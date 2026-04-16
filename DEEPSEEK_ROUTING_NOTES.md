# DeepSeek-V3 Auxiliary-Loss-Free Routing — Implementation Notes

## Scope
Ported from `olmo` repo (`train-olmoe-moe` branch) into `olmo-rocm` on 2026-04-16
to support `moe_routing_type: deepseek` ablations.

## Reference sources
- **Paper**: arXiv:2412.19437 §2.1.2 "Auxiliary-Loss-Free Load Balancing"
  (+ "Complementary Sequence-Wise Auxiliary Loss")
- **Training reference**: NVIDIA Megatron-Core
  - `megatron/core/transformer/moe/moe_utils.py` (`topk_routing_with_score_function`, `get_updated_expert_bias`)
- **Forward-pass reference**: HuggingFace `transformers/models/deepseek_v3/modeling_deepseek_v3.py`
  (`DeepseekV3TopkRouter`)
- **Official V3 inference**: `deepseek-ai/DeepSeek-V3/inference/model.py` class `Gate`

## What is implemented (`_router_deepseek_hook` in `olmo/model.py`)

| Paper spec | Status | Notes |
|---|---|---|
| Sigmoid per-expert gating | ✅ | Recomputed from router logits; overrides megablocks' softmax. |
| Bias added to scores for top-k selection only | ✅ | `biased = sigmoid(logits) + bias` |
| Gate weights from *unbiased* sigmoid scores, L1-renormalized | ✅ | `gate = sig.gather(1, idx); gate /= gate.sum(-1)` |
| Bias update `b += γ · sign(expected − actual)` | ✅ | γ = `moe_deepseek_bias_update_rate` (default 0.001, matches paper) |
| Sequence-wise complementary aux loss, α = 1e-4 | ⚠️ approximated | Computed over the whole microbatch as one "sequence" because megablocks flattens (batch, seq) before the router, losing sequence boundaries. In practice this slightly *under-counts* per-sequence imbalance vs. the paper's formulation. |

## Batch size rationale

Ablation configs use `global_train_batch_size=1024`, `device_train_microbatch_size=64` (16 accumulation steps, ~4M tokens/step). This matches the DeepSeek auxiliary-loss-free paper's 1B ablation scale (1152 sequences × 4096 tokens ≈ 4.7M tokens/step) and is required for the bias update to operate on a clean full-step signal rather than noisy microbatch fragments.

## Known gaps vs. paper / Megatron-Core

### 1. ~~Update cadence: per-microbatch, not per-step~~ — FIXED
- **Paper / Megatron**: bias update runs **once per optimizer step**, using the full-batch load.
- **Fix**: `actual_load` is now accumulated into `_deepseek_load_accum` across microbatches in the forward hook; `apply_deepseek_bias_update()` is called once per step from `train.py` after the microbatch loop, applies `sign(expected − accumulated)`, then resets the accumulator.

### 2. No cross-rank all-reduce of `actual_load`
- **Paper / Megatron**: sum `tokens_per_expert` across the global batch group (DP × TP × CP) before computing `sign(offset)`.
- **Here**: uses local-rank counts only.
- **Effect on current 1-GPU config**: **no difference** (1 rank = global batch).
- **Breaks when**: DP / FSDP across >1 rank. Bias state on each rank drifts independently → routing becomes rank-dependent → broken.
- **Fix when scaling up**: `dist.all_reduce(actual_load, group=data_parallel_group)` before the `sign()` update.

### 3. Bias state not checkpointed
- **Paper**: trains for 14.3T tokens with continuous bias updates; never tests resumption.
- **Here**: `_deepseek_bias` is a plain Python attribute, not a `register_buffer`. Same design as `_ema_mean` in the EMA path, with the same rationale: olmo_core's sharded checkpointer does not save buffers, and registering as a buffer introduces FSDP `bf16` buffer-dtype casting issues.
- **Effect on current workflow** (`load_path: null`, `try_load_latest_save: false`, `save_overwrite: true`): **no effect** — every run starts from bias = 0 and accumulates for the full run.
- **Breaks when**: you crash and resume from a sharded checkpoint, or do multi-stage training. On resume, the bias resets to 0 and the router briefly reverts to uniform before re-balancing. Mild transient effect; self-corrects in ~hundreds of steps at γ = 0.001.
- **Fix**: extend olmo_core checkpointer to save a side-car dict of per-layer deepseek bias tensors, or promote bias to an `nn.Parameter` with `requires_grad=False` and zero weight decay.

### 4. Sequence-wise aux loss is microbatch-wise, not per-sequence
- **Paper**: α · Σᵢ fᵢ · Pᵢ · N_r, computed **per sequence**, averaged across sequences.
- **Here**: computed over the whole flattened microbatch. Megablocks has already flattened `(batch, seq) → tokens` by the time the hook runs, so we cannot recover sequence boundaries without plumbing `cu_doc_lens` into the router.
- **Effect**: fewer per-sequence extremes get penalized, so the loss is a slightly weaker regularizer than the paper's. Given α = 1e-4, the practical difference is tiny.
- **Fix when it matters**: thread `cu_doc_lens` into the hook and split per sequence before computing `f_i`, `P_i`.

### 5. No `route_scale` multiplier on final gate weights
- **Paper / V3 code**: `weights *= route_scale` after L1 renormalization.
- **Here**: omitted.
- **Effect**: `route_scale` is a model-specific hyperparameter calibrated for the 671B V3 model (tied to the `dim=7168` branch in the V3 `Gate` code). For the 1.2B-active OLMoE-scale model we're running, there is no paper-specified value. Omission is consistent with the existing `olmo` repo port and with Megatron-Core's default behavior.

## Comparison table

| Aspect | Paper | Megatron-Core | This port (1 GPU) |
|---|---|---|---|
| Gating activation | sigmoid | sigmoid | sigmoid ✅ |
| Bias → selection only | ✅ | ✅ | ✅ |
| Gate weights = unbiased + renorm | ✅ | ✅ | ✅ |
| Bias update rule | `sign(expected−actual) · γ` | ✅ | ✅ |
| γ default | 0.001 | 0.001 | 0.001 ✅ |
| Update cadence | per step | per step | per step ✅ |
| Cross-rank all-reduce | global batch | TP×DP×CP group | not done (N/A on 1 GPU) |
| Seq aux loss α | 1e-4 | optional (same α) | 1e-4, microbatch-wise approximation |
| Bias checkpointing | N/A (no resume) | saved in training state | not saved |
| `route_scale` | model-specific | passthrough | not applied |

## When to revisit
Before any of the following, re-read this file and fix the corresponding gap:
- Running on > 1 GPU → fix gap 2 (all-reduce)
- Resuming from checkpoint → fix gap 3 (checkpoint bias)
- Claiming strict paper parity in a write-up → fix gaps 2, 3, 4
