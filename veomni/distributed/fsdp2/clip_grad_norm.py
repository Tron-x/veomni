import math
import os
import re
from typing import Dict, List

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

from ...utils.device import get_device_type
from ...utils.logging import get_logger
from ..parallel_state import get_parallel_state


logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Optional per-layer grad-norm dump for numerical-alignment debugging.
#
# Enabled by env var VEOMNI_GRAD_DUMP=<path> — when set, every call to
# ``extra_parallel_fsdp2_clip_grad_norm`` buckets parameters by
# ``transformer-block index`` (extracted from ``named_parameters()`` name)
# and ``param group`` (``non_extra_parallel`` / ``ep`` / ``emb`` / ...),
# then performs ONE all-reduce per bucket to produce a globally-reduced
# L2 norm. Output format (one line per bucket):
#
#   STEP=<step> LAYER=<block_idx_or_'other'> GROUP=<name> NORM=<float>
#
# This is intentionally NOT integrated into clip_grad_norm's reduction
# tree — it adds extra communication and floating-point work, so MUST be
# gated by env var. Used to:
#   (a) compare per-layer global grad norm between two distributed
#       topologies (e.g. 1node-8NPU vs 2node-16NPU) to find the layer
#       where divergence first appears;
#   (b) confirm whether divergence concentrates in MoE layers (=> EP
#       routing/EP-FSDP sharding effect) vs attention layers (=> FSDP
#       reduce-scatter ordering effect).
#
# Step number is tracked via a module-level counter that increments each
# time this clipper is invoked (one invocation = one training step).
# ------------------------------------------------------------------ #
_GRAD_DUMP_STEP = 0
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    # ExtraParallel-aware path (FSDP2 + ExtraParallel): maintain mathematical parity with FSDP1 clipper

    if hasattr(model, "_extra_parallel_param_groups"):
        return extra_parallel_fsdp2_clip_grad_norm(
            model,
            max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        )

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
        foreach=foreach,
    )
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return grad_norm


@torch.no_grad()
def extra_parallel_fsdp2_clip_grad_norm(
    model, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False, foreach: bool | None = None
) -> torch.Tensor:
    """
    ExtraParallel-aware gradient clipping for composable FSDP2 with reductions mirroring FSDP1:

    - Compute local norms for non-ExtraParallel and ExtraParallel parameter groups separately.
    - For finite p: sum p-th powers across the appropriate groups, then take 1/p.
      • non-ExtraParallel: all-reduce over FSDP group.
      • ExtraParallel: all-reduce over Para-FSDP (e.g. ep_fsdp, emb_fsdp) group, then over Para (e.g. ep, emb) group.
    - For inf-norm: take elementwise MAX with the same reduction groups (MAX).
    - Use a single global clip coefficient for both groups.
    """
    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
    extra_parallel_group = {
        para: ps.extra_parallel_group(para) if ps.extra_parallel_enabled(para) else None
        for para in ps.extra_parallel_names
    }
    # For Para (e.g. ep, emb) params sharded by FSDP2 along hidden dimension
    extra_parallel_fsdp_group = {
        para: ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
        if ps.extra_parallel_enabled(para) and ps.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in ps.extra_parallel_names
    }

    # Build param groups for ExtraParallel params and non-ExtraParallel params (filter out params without grads)
    extra_parallel_params = {
        para: [p for p in model._extra_parallel_param_groups.get(para, []) if p.grad is not None]
        for para in ps.extra_parallel_names
    }
    non_extra_parallel_params: List[torch.nn.Parameter] = [
        p for p in model._extra_parallel_param_groups.get("non_extra_parallel", []) if p.grad is not None
    ]

    # Compute and reduce non-ExtraParallel
    non_extra_parallel_total = _fsdp2_reduce_group(
        params=non_extra_parallel_params,
        norm_type=norm_type,
        reduce_groups=[("fsdp", fsdp_group)],
    )
    logger.debug_rank0(f"non_extra_parallel total grad norm: {non_extra_parallel_total}")

    for para in ps.extra_parallel_names:
        logger.debug_rank0(
            f"{para}_params reduces groups: {extra_parallel_fsdp_group[para]=}, {extra_parallel_group[para]=}"
        )

    # Compute and reduce ExtraParallel: first across para_fsdp (e.g. ep_fsdp, emb_fsdp), then across para (e.g. ep, emb)
    extra_parallel_total = {
        para: torch.tensor(0.0, device=torch.device(get_device_type()), dtype=torch.float32)
        for para in ps.extra_parallel_names
    }
    for para in ps.extra_parallel_names:
        if len(extra_parallel_params[para]) > 0:
            para_total = _fsdp2_reduce_group(
                params=extra_parallel_params[para],
                norm_type=norm_type,
                reduce_groups=[
                    (f"{para}_fsdp", extra_parallel_fsdp_group[para]),
                    (f"{para}", extra_parallel_group[para]),
                ],
            )
            extra_parallel_total[para] = para_total
            logger.debug_rank0(f"{para} total grad norm: {para_total}")

    if math.isinf(norm_type):
        total_norm = torch.maximum(non_extra_parallel_total, *extra_parallel_total.values())
    else:
        total_norm = (non_extra_parallel_total + sum(extra_parallel_total.values())) ** (1.0 / float(norm_type))

    # Optional per-layer grad-norm dump — MUST run BEFORE clip_grads_with_norm_
    # so the per-layer values are PRE-clip (i.e. directly comparable to the
    # ``grad_norm`` printed by the trainer for the same step). If we dumped
    # after clipping, every layer's value would be scaled by
    # ``min(max_norm / total_norm, 1.0)``, which changes across runs (since
    # total_norm itself differs across the topologies we're comparing) and
    # would mix two distinct effects in the diff.
    dump_path = os.environ.get("VEOMNI_GRAD_DUMP", "").strip()
    if dump_path:
        _dump_per_layer_grad_norms(model, norm_type=norm_type, dump_path=dump_path)

    # Apply the same clip coefficient to both groups
    for para in ps.extra_parallel_names:
        torch.nn.utils.clip_grads_with_norm_(extra_parallel_params[para], max_norm, total_norm, foreach=foreach)
    torch.nn.utils.clip_grads_with_norm_(non_extra_parallel_params, max_norm, total_norm, foreach=foreach)

    return total_norm


def _classify_param_group(model, p: torch.nn.Parameter) -> str:
    """Return which ``_extra_parallel_param_groups`` bucket ``p`` belongs to."""
    groups = getattr(model, "_extra_parallel_param_groups", None)
    if groups is None:
        return "unknown"
    for name, params in groups.items():
        for q in params:
            if q is p:
                return name
    return "unknown"


def _extract_layer_key(name: str) -> str:
    """Bucket key: transformer-block index if present, else 'other'."""
    m = _LAYER_RE.search(name)
    return f"layer{int(m.group(1)):02d}" if m else "other"


@torch.no_grad()
def _dump_per_layer_grad_norms(model, norm_type: float, dump_path: str) -> None:
    """Compute and log globally-reduced per-(layer,group) L2 grad norms.

    Communication cost: ONE all-reduce per non-empty bucket. With Pangu
    Omni v2 (27 transformer layers) and 2 param groups (non_extra_parallel
    and ep), this is at most ~54 small scalar all-reduces per step — fine
    for short alignment runs but absolutely NOT for production.
    """
    global _GRAD_DUMP_STEP
    _GRAD_DUMP_STEP += 1
    step = _GRAD_DUMP_STEP

    if math.isinf(norm_type):
        return  # inf-norm path not implemented for the dump; not needed.
    p = float(norm_type)

    ps = get_parallel_state()
    fsdp_group = ps.fsdp_group
    extra_parallel_group = {
        para: ps.extra_parallel_group(para) if ps.extra_parallel_enabled(para) else None
        for para in ps.extra_parallel_names
    }
    extra_parallel_fsdp_group = {
        para: ps.extra_parallel_fsdp_device_mesh[para][f"{para}_fsdp"].get_group()
        if ps.extra_parallel_enabled(para) and ps.extra_parallel_fsdp_device_mesh[para] is not None
        else None
        for para in ps.extra_parallel_names
    }

    buckets: Dict[tuple[str, str], List[torch.nn.Parameter]] = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        layer_key = _extract_layer_key(name)
        group_key = _classify_param_group(model, param)
        buckets.setdefault((layer_key, group_key), []).append(param)

    # Different EP ranks may not route tokens to the same experts, so some
    # buckets can be locally empty. Every rank must still issue collectives in
    # the same order, reducing a zero tensor for missing buckets.
    bucket_keys = sorted(buckets)
    if dist.is_initialized():
        gathered_keys = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_keys, bucket_keys)
        bucket_keys = sorted({key for rank_keys in gathered_keys for key in rank_keys})

    is_rank0 = dist.get_rank() == 0 if dist.is_initialized() else True
    fh = None
    if is_rank0:
        fh = open(dump_path, "a")

    for layer_key, group_key in bucket_keys:
        params = buckets.get((layer_key, group_key), [])
        local_sum = _local_pth_sum(params, p)
        if group_key in ps.extra_parallel_names:
            efsdp = extra_parallel_fsdp_group.get(group_key)
            if efsdp is not None:
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=efsdp)
            egroup = extra_parallel_group.get(group_key)
            if egroup is not None:
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=egroup)
        else:
            if fsdp_group is not None:
                dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=fsdp_group)

        layer_norm = local_sum.pow(1.0 / p).item()
        if fh is not None:
            fh.write(f"STEP={step} LAYER={layer_key} GROUP={group_key} NORM={layer_norm:.8e}\n")

    if fh is not None:
        fh.close()


# compute local sum of param gard norm
def _local_pth_sum(params: List[torch.nn.Parameter], p: float) -> torch.Tensor:
    grads = [p.grad for p in params if p.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor) else g.detach().to(torch.float32)
        for g in grads
    ]

    default_device = grads_local[0].device if len(grads_local) > 0 else torch.device(get_device_type())
    res = torch.tensor(0.0, device=default_device, dtype=torch.float32)
    with torch.no_grad():
        grouped_grads_local = _group_tensors_by_device_and_dtype([grads_local])
        for (device, _), ([device_grads_local], _) in grouped_grads_local.items():
            if _has_foreach_support(device_grads_local, device) or _device_has_foreach_support(device):
                out = torch._foreach_pow_(torch._foreach_norm(device_grads_local, p), p)
                res += torch.sum(torch.stack(out)).to(default_device)
            else:
                for grad_local in device_grads_local:
                    gn = torch.norm(grad_local, p=p)
                    res = res + (gn**p).to(default_device)
    return res


def _local_max(params: List[torch.nn.Parameter]) -> torch.Tensor:
    dev = None
    mx = None
    for q in params:
        g = q.grad
        if g is None:
            continue
        if isinstance(g, DTensor):
            g_local = g.to_local()
        else:
            g_local = g
        if dev is None:
            dev = g_local.device
            mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
        gn = torch.max(torch.abs(g_local.detach().to(torch.float32)))
        mx = torch.maximum(mx, gn)
    if mx is None:
        dev = torch.device(get_device_type())
        mx = torch.tensor(0.0, device=dev, dtype=torch.float32)
    return mx


def _fsdp2_reduce_group(
    params: List[torch.nn.Parameter],
    norm_type: float,
    reduce_groups: List[tuple[str, dist.ProcessGroup | None]],
) -> torch.Tensor:
    """Compute local group statistic and reduce over provided groups.

    For finite p, returns the globally-reduced sum of p-th powers (not the final norm).
    For inf, returns the globally-reduced max.
    """
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        logger.debug_rank0(f"local total grad norm: {val}. ProcessGroups to sum {reduce_groups}")
        for name, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
                logger.debug_rank0(f"After Sum of group {name} total grad norm is {val}")
        return val
