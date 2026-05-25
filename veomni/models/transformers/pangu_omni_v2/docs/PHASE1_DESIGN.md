# Pangu Omni 30B-A2B → VeOmni 适配设计 (Phase 1)

> 状态：草案 v0.1 | 2026-05-21 | 基于 fail-fast load 实测 + VeOmni 深扫综合
>
> 这份文档不是"理论评估"。文档里**所有 gap 和约束都来自实跑验证或 grep
> 实证**——见 `tools/pangu_veomni_failfast_load.py` 输出和 modeling 代码扫描。

---

## 0. 目标与硬约束

### 0.1 业务目标

把 **Pangu Omni 30B-A2B**（华为盘古团队多模态 MoE 模型）适配进 **VeOmni** 训练
框架，实现 NPU 上的端到端 SFT/RL 训练。

### 0.2 硬约束（来自用户明确要求）

1. **跟 VeOmni 主仓共仓**——我们的代码要能被上游接收
2. **盘古扩展不污染主仓**——非 Pangu 用户感知不到我们的代码
3. **GPU/NPU 共精度**——同一份 modeling 代码 GPU/NPU 双跑数值一致
4. **支持后续多代多种盘古模型**（v2 → v3 → ...，文本/VL/Omni）

这 4 条约束是**所有设计决策的判据**——见 §5 "为什么我们这样设计"。

---

## 1. Step 1 fail-fast load 实测结果

跑了 `tools/pangu_veomni_failfast_load.py`，两个模式（`MODELING_BACKEND=hf` 和默认
`veomni`）的探针结果如下：

### 1.1 `MODELING_BACKEND=hf` 模式

| Probe | 结果 | 关键细节 |
|---|---|---|
| `build_config` | ✅ pass | `OpenPanguOmniConfig` 加载成功，但 2 个 warning（见 §2.1, §2.4） |
| `get_model_class` | ✅ pass | 解析到 `transformers.AutoModel`（HF 自带 fallback） |
| `build_foundation_model` | ❌ **fail** | `trust_remote_code` 被 VeOmni 内部 pop 后没透传，HF 拒绝执行 Pangu 自带 modeling 代码 |

### 1.2 `MODELING_BACKEND=veomni` 模式（默认）

| Probe | 结果 | 关键细节 |
|---|---|---|
| `build_config` | ✅ pass | 同上 |
| `get_model_class` | ❌ **fail** | `MODELING_REGISTRY` 里没有 `qwen2_moe` 注册（model_type 字符串不对，见 §2.1） |

**结论**：Pangu 现状是**两条路都走不通**。这正是 phase 1 要解决的事。

---

## 2. 已发现的 5 个 Gap（按严重度排序）

### 2.1 🟠 **Gap A**：`model_type = "qwen2_moe"` 字符串错配

`config.json:`
```json
{
  "model_type": "qwen2_moe",
  "architectures": ["OpenPanguUltraOmniForConditionalGeneration"]
}
```

`model_type` 是 HF transformers 的注册键，但盘古直接复用了 `qwen2_moe` 字符串。
后果：

- transformers 5.0.0 warning：`"You are using a model of type qwen2_moe to instantiate a model of type openpangu_omni"`
- VeOmni `MODELING_REGISTRY` 试图按 `qwen2_moe` 查找注册，找不到 → 上面 1.2 的 fail

**两种解法**：

| 选项 | 优点 | 缺点 | 推荐 |
|---|---|---|---|
| 让盘古团队改 config.json `model_type` 为 `openpangu_omni` | 干净；符合 HF 规范 | 需要协调上游团队 + 已发布 ckpt 都要改 | ✅ 推荐（**短期**：我们 adapter 加 config patch 自动改字段） |
| 我们 adapter 注册 `qwen2_moe` 但只在 `architectures` 含 OpenPangu 时启用 | 不改 ckpt | 污染 Qwen2 路径；架构混乱；主仓肯定拒收 | ❌ |

**Phase 1 落地**：在 `MODEL_CONFIG_REGISTRY` 注册时 patch `model_type` 字段，
对外仍兼容 `qwen2_moe` 但内部走 `openpangu_omni` 路径。

### 2.2 🔴 **Gap B**：VeOmni 不支持 HF `trust_remote_code` 透传（架构性）

VeOmni `loader.py:191`：
```python
init_kwargs.pop("trust_remote_code", True)  # ← 直接 pop 掉，不透传
```

这是 VeOmni 的**架构假设**："所有支持的模型都已 mainline 到 transformers"。
对盘古这种**带 custom modeling code 的模型**是个 blocker。

**含义**：我们 adapter 必须**把 Pangu modeling 代码"内化"到 VeOmni package 里**，
不能依赖 HF 的 dynamic loading 机制。这是 phase 1 最大的工作量来源。

### 2.3 🟠 **Gap C**：modeling 代码本身的 GPU 不兼容（违反约束 0.2-3）

| 文件 | 行 | 问题 | 影响 |
|---|---|---|---|
| `modeling_openpangu_vl.py` | 28 | **无条件** `import torch_npu` | GPU 机器一 import 就爆 |
| `modeling_openpangu_vl.py` | 55-57 | conditional `if is_torch_npu_available()`（已正确） | OK |
| `modeling_pangu_omni.py` | 58-60 | conditional 同上 | OK |
| `processor_openpangu_omni.py` | 51-80 | `ctypes.cdll.LoadLibrary('FeatureExteract.old.so')`——华为内部二进制 | GPU 机器没这个 .so |

**含义**：
- 28 行的 unconditional `import torch_npu` **必须改成 try/except**（盘古上游 bug，
  我们 adapter 落地时打 patch 修掉）
- `FeatureExteract.so` 做的事是 **fbank40 特征提取**（标准 Mel filterbank），
  可以用 `torchaudio.compliance.kaldi.fbank` PyTorch 纯实现**完全等价**替代
  → 这是 GPU fallback 的标准做法

### 2.4 🟡 **Gap D**：`rope_parameters.rotary_mode` 是华为内部字段

transformers 5.0.0 warning：
```
Unrecognized keys in `rope_parameters` for 'rope_type'='default': {'rotary_mode'}
```

`rotary_mode` 是盘古 partial RoPE 的控制字段（`partial_rotary_factor=0.25`,
`qk_rope_dim=32`）。transformers mainline 不认。

**含义**：rotary embedding 模块需要 VeOmni patch（这本来就在 phase 1 任务清单
里——MHC / partial RoPE / K-norm 都是 Pangu 独有）

### 2.5 🟢 **Gap E**：NPU validator 工作正常（已解决）

VeOmni 的 NPU validator 自动拒绝 `liger_kernel`，建议 `eager`/`npu`。这是**好消息**——
说明 VeOmni 已经认真做了 NPU 适配框架，我们只需正确选 ops_implementation。

---

## 3. NPU-specific 代码清单（GPU/NPU 共精度的工作面）

| 位置 | 性质 | GPU fallback 策略 |
|---|---|---|
| `modeling_openpangu_vl.py:28` `import torch_npu` | 无条件 import bug | Patch 改 try/except |
| `modeling_openpangu_vl.py:55-57` NPU attention infer | conditional + 标记 `NPU_ATTN_INFR` | 已经是 NPU/GPU 分支模式，无需改 |
| `modeling_pangu_omni.py:58-60` 同上 | conditional | 同上 |
| `processor_openpangu_omni.py:51` `FeatureExteract.old.so` | 二进制 .so | 用 `torchaudio.compliance.kaldi.fbank` 等价替代 |
| `processor_openpangu_omni.py:55-56` `mean.npy / var.npy` | 数据文件（norm 参数） | 复用——文本 .npy 不绑硬件 |

**好消息**：`modeling_openpangu_v2.py`（text MoE backbone，最复杂）**完全 device-agnostic**——
没找到任何 `torch_npu` / `.so` 调用。这意味着 MHC、partial RoPE、K-norm 这些
Pangu 独有的核心算法**天然 GPU/NPU 双跑**。

---

## 4. 适配方案：三个候选 + 推荐

基于 Gap B（VeOmni 不支持 trust_remote_code），我们必须把 Pangu modeling 代码
内化进 adapter package。"内化"有 3 种方案：

### 方案 P1：**完全拷贝 + v5 patchgen 化**（推荐）

把 `configs/modeling_*.py` 三个文件拷到 `veomni/models/transformers/pangu_omni_v2/`，
按 qwen3_omni_moe 模板做 v5 patchgen 化。

| 维度 | 评估 |
|---|---|
| 工作量 | 中（3-4 周第一款，后续款 0.5-2 周） |
| 与 VeOmni 主仓兼容 | ✅ 完全符合主仓约定 |
| 盘古上游同步 | 中等——盘古团队改 modeling 时我们要 sync |
| GPU/NPU 共精度 | ✅ 可控 |
| 多代支持 | ✅ 通过 `_pangu_common/` 共享层 |

### 方案 P2：**盘古 modeling 上游到 transformers mainline**

短期不现实。盘古团队对 transformers 的依赖深度（华为 fork transformers / MindSpeed
深度耦合）使 upstream 路径非常长。**但作为长期目标推动**。

### 方案 P3：**给 VeOmni 加 trust_remote_code 通道**

改 VeOmni `loader.py` 透传 `trust_remote_code=True`。

| 维度 | 评估 |
|---|---|
| 工作量 | 小（VeOmni 30 行修改） |
| 与主仓兼容 | ❓ 需要 VeOmni 上游 review；可能拒收（违反 "all models mainline" 假设） |
| 多代支持 | ❌ 每个 Pangu 新型号都要重新带一份 modeling code 到 ckpt 里 |
| 维护成本 | 高——盘古团队 modeling 改动直接影响生产 |

**推荐：方案 P1**（主路径）+ 方案 P3 作为开发期 dev 工具（**不上游**，本地用）+
方案 P2 作为长期目标。

---

## 5. 共享层设计：`_pangu_common/`

为支持后续多代多型号盘古（约束 0.2-4），抽出 `_pangu_common/`。

### 5.1 目录结构

```
veomni/models/transformers/
├── _pangu_common/                      # ← Pangu 系列共享层（新增）
│   ├── __init__.py
│   ├── README.md                       # 说明：本目录代码仅 Pangu 系列使用
│   ├── pangu_mhc.py                    # MHC: 4-stream + gamma scaling + sinkhorn-knopp
│   ├── pangu_partial_rope.py           # partial RoPE + qk_rope_dim 配置驱动
│   ├── pangu_knorm.py                  # K-norm，注册为 VeOmni OpSlot
│   ├── pangu_vision_gated_merger.py    # GatedMerger（vs Qwen3-Omni PatchMerger）
│   ├── pangu_moe_shared_experts.py     # MoE + 2 shared experts 实现
│   ├── pangu_huanyu_audio.py           # HuanyuAudioEncoder PyTorch 实现
│   ├── pangu_fbank_extract.py          # fbank40 PyTorch (替代 FeatureExtract.so)
│   ├── parallel_plan_templates.py      # 通用 EP/FSDP 模板，子模型继承覆盖
│   └── checkpoint_converter_base.py    # MoE 权重转换基类
│
├── pangu_omni_v2/                      # ← 当前 30B-A2B（第一款）
│   ├── __init__.py                     # 注册 model_type
│   ├── configuration_pangu_omni_v2.py  # config 子类，patch model_type 字符串
│   ├── modeling_pangu_omni_v2.py       # v4 monkey-patch（小，主要是装饰 + 桥接 _pangu_common）
│   ├── pangu_omni_v2_gpu_patch_gen_config.py  # v5 patchgen 意图（引用 _pangu_common）
│   ├── generated/patched_modeling_pangu_omni_v2_gpu.py  # patchgen 自动生成
│   ├── parallel_plan.py                # 继承 _pangu_common.parallel_plan_templates
│   ├── checkpoint_tensor_converter.py  # 继承 _pangu_common.checkpoint_converter_base
│   └── README.md
│
├── pangu_omni_v3/                      # ← 未来：下一代 Omni（占位）
│   └── README.md                       # "TBD — 90% 复用 pangu_omni_v2 + _pangu_common"
│
└── ...                                  # 其他 VeOmni 已支持模型
```

### 5.2 复用度估算

| 后续型号情景 | 工作量 | 复用率 |
|---|---|---|
| Pangu Omni v2 70B-A7B（同 model_type，只改 config 规模） | **0 代码 + 0.5-1 天验证** | 100% |
| Pangu Text v2（同代纯文本，model_type 变） | 1-2 周 | 80% |
| Pangu VL v2（同代纯视觉） | 1-2 周 | 80% |
| Pangu Omni v3（下一代，新增功能） | 2-3 周 | 60% |
| Pangu Omni v3 with 新模态（视频生成等） | 3-5 周 | 40% |

第一款（pangu_omni_v2）成本：**3-4 周**。

### 5.3 与 VeOmni 主仓的关系

- `_pangu_common/` 是新增目录，主仓代码完全不感知
- `pangu_omni_v2/` 是新增 package，按 VeOmni 现有约定（每个 model_type 一个包）
- **触碰主仓代码的总量 = 8 行注册**（详见 §6.2）
- 这种隔离度让上游 review 阻力最小

---

## 6. Phase 1 工作分解（4-6 周）

### 6.1 文件创建清单

| # | 文件 | 行数估算 | 工作量 |
|---|---|---|---|
| 1 | `_pangu_common/pangu_mhc.py` | 400 | 大 |
| 2 | `_pangu_common/pangu_partial_rope.py` | 100 | 小 ✅ DONE (W1D1) |
| 3 | `_pangu_common/pangu_rms_norm.py` (was pangu_knorm.py — K-norm is just RMSNorm at key proj, no separate op needed) | 50 | 小 |
| 4 | `_pangu_common/pangu_attention.py` (K-norm + partial RoPE wiring in attention forward) | 250 | 中 |
| 5 | `_pangu_common/pangu_vision_gated_merger.py` | 150 | 中 |
| 6 | `_pangu_common/pangu_moe_shared_experts.py` | 300 | 中 |
| 7 | `_pangu_common/pangu_huanyu_audio.py` | 400 | 大 |
| 8 | `_pangu_common/pangu_fbank_extract.py` | 100 | 小 |
| 9 | `_pangu_common/parallel_plan_templates.py` | 80 | 小 |
| 10 | `_pangu_common/checkpoint_converter_base.py` | 150 | 小 |
| 10 | `pangu_omni_v2/__init__.py` | 100 | 小 |
| 11 | `pangu_omni_v2/configuration_pangu_omni_v2.py` | 80 | 小 |
| 12 | `pangu_omni_v2/modeling_pangu_omni_v2.py` (v4) | 800 | 中 |
| 13 | `pangu_omni_v2/pangu_omni_v2_gpu_patch_gen_config.py` (v5) | 600 | 中 |
| 14 | `pangu_omni_v2/parallel_plan.py` | 40 | 小 |
| 15 | `pangu_omni_v2/checkpoint_tensor_converter.py` | 100 | 小 |
| **总计** | **~3500 行手写代码** + **~3000 行 patchgen 自动生成** | | |

### 6.2 主仓必须 touch 的点（共 8 处）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `veomni/models/transformers/__init__.py` | `from . import _pangu_common, pangu_omni_v2` |
| 2 | `veomni/data/data_transform.py` | `@DATA_TRANSFORM_REGISTRY.register("openpangu_omni")` 或复用 omni 通用 |
| 3 | `veomni/trainer/vlm_trainer.py` | `model_type in (..., "openpangu_omni")` 加到 omni 分支 |
| 4 | `configs/multimodal/pangu_omni/pangu_omni_v2_30b_a2b.yaml` | 新增训练配置 |
| 5 | `tests/toy_config/pangu_omni_v2_toy/config.json` | 新增 toy 测试配置 |
| 6 | `tests/models/test_models_logits_equal_v5.py` | 加 pangu 用例 |
| 7 | `tests/models/test_checkpoint_tensor_converter.py` | 加 pangu 用例 |
| 8 | `docs/usage/support_new_models/pangu_omni_example.md` | 新增文档 |

### 6.3 按周拆分

#### Week 1：基础设施 + 文本 MoE backbone
- Day 1-2：建 `_pangu_common/` 目录骨架；写 `parallel_plan_templates` / `checkpoint_converter_base`
- Day 3-4：实现 `pangu_partial_rope.py` + `pangu_knorm.py`
- Day 5：实现 `pangu_moe_shared_experts.py`（不含 MHC）
- **里程碑 1**：单步 forward 在 dense layer（前 2 层）logp 跟 HF oracle 对齐 < 1e-4

#### Week 2：MHC + 完整文本路径
- Day 1-3：实现 `pangu_mhc.py`（最复杂，需要数值对齐 sinkhorn-knopp）
- Day 4-5：拼装 `pangu_omni_v2/modeling_pangu_omni_v2.py`（v4 路径，先跑通）
- **里程碑 2**：单 sample 全 37 层 forward logp 对齐 OCRBench 任一样本 < 1e-3

#### Week 3：视觉 + 音频
- Day 1-2：`pangu_vision_gated_merger.py` + ViT 适配
- Day 3-4：`pangu_huanyu_audio.py` + `pangu_fbank_extract.py`（用 torchaudio）
- Day 5：联调，跑 OCRBench 多模态 sample
- **里程碑 3**：MMMU 一个 sample 跑通且 logp 对齐 < 1e-3

#### Week 4：v5 patchgen 化 + EP + 集成测试
- Day 1-2：写 `pangu_omni_v2_gpu_patch_gen_config.py`，跑 patchgen
- Day 3-4：实现 EP `parallel_plan.py`，单机多卡 EP 跑通
- Day 5：写 toy config + parity test
- **里程碑 4**：8 卡 NPU EP=8 跑通，全 OCRBench 1000 样本 logp 平均差异 < 1e-3

#### Week 5-6：缓冲（处理 unknown unknowns）
- 缓冲 + checkpoint 转换 + 数据 pipeline 集成 + 文档 + 上游 PR 准备

### 6.4 验证标准（每周里程碑共用）

每个里程碑都用 **`tools/pangu_oracle_check.py`** 跑：
```python
# 输入: VeOmni 模型实例 + N 条 OCRBench 样本
# 步骤:
#   1. 用 HF AutoModel + trust_remote_code 跑一遍 → (text_pred, token_ids_hf, logps_hf)
#   2. 用 VeOmni 模型跑相同输入 → (token_ids_ve, logps_ve)
#   3. assert token_ids_hf == token_ids_ve (greedy decoding 必须完全一致)
#   4. assert max(abs(logps_hf - logps_ve)) < tolerance
# 输出: per-token max/mean diff, fail rate
```

Oracle 已就绪：`/mnt/data_3/models/pangu/test_hf_percision.0518.parallel/` 里
`run_hf_one` 函数 + `results/ocrbench_hf_outputs.jsonl` 1000 条 baseline。

---

## 7. 关键风险 + 缓解策略

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| MHC 数值对齐失败（sinkhorn-knopp 数值不稳） | 中 | 大 | 周 1 就开始原型，留双倍缓冲 |
| HuanyuAudio + torchaudio fbank 数值差异 | 中 | 中 | 先用 .so 跑通完整 pipeline，再替换 fbank 实现，diff 隔离 |
| `FeatureExteract.so` 输出 vs torchaudio fbank 算法不完全等价 | 低 | 小 | 可以保留 .so 作为 NPU 优化路径，GPU 用 torchaudio |
| v5 patchgen 在 Pangu 复杂 modeling 上跑不通 | 低 | 中 | 先做 v4 monkey-patch 兜底；v5 patchgen 是优化项 |
| VeOmni 上游 review 拒收 `_pangu_common/` 目录 | 中 | 大 | 先内部跑通；review 沟通时强调"完全隔离、不影响主仓功能" |
| Partial RoPE + MRoPE 组合的位置编码错位 | 中 | 大 | 周 1 单独写 unit test，固定一个种子比较 cos/sin 表 |
| 30B 模型单卡装不下，必须先 EP 才能验证 logp | 高 | 中 | 用 toy config（缩小到 6 层 + 16 experts）验证算法，最后再大模型集成 |

---

## 8. 决策点（已确认 2026-05-21）

| # | 决策 | **决议** | 备注 |
|---|---|---|---|
| D1 | model_type 字符串处理 | ✅ **adapter 透明 patch** | 不依赖盘古团队改 config.json；adapter 内部把 `qwen2_moe` 映射成 `openpangu_omni` |
| D2 | v4/v5 双轨 vs 只做 v5 | ✅ **只做 v5 patchgen** | transformers 5.0.0 锁定，砍 v4 节省 ~1500 行；唯一 modeling 实现路径 |
| D3 | `_pangu_common/` 位置 | ✅ **`veomni/models/transformers/_pangu_common/`** | 与 VeOmni 现有 package 约定一致；下划线前缀提示"非 model_type 包" |
| D4 | HuanyuAudio fbank 实现 | ✅ **双路** | NPU 默认 `.so` (性能)，GPU 默认 torchaudio (兼容)；环境变量 `PANGU_FBANK_BACKEND={so,torchaudio,auto}` 切换 |
| D5 | 是否同期推 VeOmni 上游 trust_remote_code 通道 | ✅ **暂不** | 方案 P1 是主路径；P3 风险高且依赖 VeOmni 上游 review 节奏 |
| D6 | 第一款验证目标 | ✅ **toy 优先** | toy: 6 层 + 16 expert (~500M)，单卡可调试；通后再上 30B-A2B 8 卡 EP |

---

## 9. 下一步立即动作（待用户审完本文档后）

1. **Day 0**（半天）：用户 review 本文档 + 回答 §8 决策点 D1-D6
2. **Day 1**（半天）：写 `tools/pangu_oracle_check.py`（phase 1 金本位检查器）
3. **Day 1-2**：搭 `_pangu_common/` 目录骨架 + `pangu_omni_v2/__init__.py`
4. **Day 3 起**：按 §6.3 周计划执行

---

## 附录 A：fail-fast load 跑通的环境快照

```
环境: monarch_ascend (conda)
Python: 3.11.0
transformers: 5.0.0
torch: 2.9.0+cpu
torch_npu: 2.9.0
huggingface_hub: 1.15.0
qwen-omni-utils: 0.0.9

模型: /mnt/data_3/models/pangu/pangu_omini_30ba2_hf_model/
  (16 个 config/code symlink → ../configs/, 2 个 safetensors 共 55GB)

VeOmni: /root/VeOmni/  (本地 checkout)
```

## 附录 B：oracle baseline 现状

| 数据集 | 样本数 | 状态 | 路径 |
|---|---|---|---|
| OCRBench | 1000 | ✅ 完整跑完 | `results/ocrbench_hf_outputs.jsonl` |
| MMMU | 11 / 100 | ⚠️ 部分（4 shards 未合并） | `results/mmmu_hf_outputs.jsonl.shards/` |

每条样本包含 `token_ids` + `tokens` + `logps`（per-token log-prob）+ `sum_logp` +
`mean_logp`——这就是数值对齐的金标准。

## 附录 C：参考文档

- VeOmni 模型注册机制深扫报告：见本次会话 explore subagent 输出
- VeOmni qwen3_omni_moe 文件级解析：见本次会话 explore subagent 输出
- VeOmni 官方新模型指南：`/root/VeOmni/.agents/skills/veomni-new-model/SKILL.md`
- VeOmni v5 patchgen 文档：`/root/VeOmni/docs/transformers_v5/patchgen.md`
- VeOmni 多模态适配示例：`/root/VeOmni/docs/usage/support_new_models/qwen3_omni_moe_example.md`
- Pangu HF 推理 oracle（已跑通）：`/mnt/data_3/models/pangu/test_hf_percision.0518.parallel/`
