# OLMoE on AMD ROCm — torch 2.9.1+rocm6.4 (legacy fallback)

Legacy reproduction steps for `torch==2.9.1+rocm6.4`. Use this only if the
recommended path in `README_ROCM.md` (`torch==2.11.0+rocm7.2`) doesn't work
on your environment.

Tested on:
- AMD Instinct MI325X (`gfx942`)
- ROCm 6.4.3 (host + torch wheel both targeting rocm6.4)
- Python 3.12, `uv` for env management
- `torch==2.9.1+rocm6.4`, `pytorch-triton-rocm==3.5.1`

> **⚠️ Why this is a fallback, not the recommended path**: triton-rocm 3.5.1
> (which ships with `torch==2.9.1+rocm6.4`) crashes with a GPU memory access
> fault inside `stk`'s sparse triton kernels when running dropless dMoE on
> MI325X. To work around it, this path is forced to use `moe_dropless: false`
> (capacity-based MoE), which **changes training dynamics** vs. the CUDA
> dropless path. The newer `torch==2.11.0+rocm7.2` path (with triton-rocm 3.6)
> fixes the crash and lets you keep dropless on. Prefer the main README unless
> you have a reason to pin to ROCm 6.4 wheels specifically.

## 1. Clone into a new directory

```bash
cd /home/morg/students/sagiahrac/repos
git clone --branch train-olmoe-moe https://github.com/sagearc/olmo.git olmo-rocm
cd olmo-rocm
```

## 2. Create env + install PyTorch ROCm 6.4

```bash
uv venv --python 3.12
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/rocm6.4
```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
# expected: 2.9.1+rocm6.4 6.4.43484-123eb5128 8
```

## 3. Pin torch in pyproject.toml (optional)

The committed `pyproject.toml` pins `torch==2.11.0`. If you're running the
legacy path, override locally so `uv pip install -e .` doesn't try to upgrade:

```bash
sed -i 's/torch==2.11.0/torch==2.9.1/' pyproject.toml
```

(Don't commit this change — keep `2.11.0` as the project default.)

## 4. Install megablocks from the sagearc fork

Same fork as the main path:
[`sagearc/megablocks-rocm@rocm-histogram-hipify-fix`](https://github.com/sagearc/megablocks-rocm/tree/rocm-histogram-hipify-fix).

```bash
uv pip install --no-build-isolation --no-deps stanford-stk==0.7.1
ROCM_HOME=/opt/rocm-6.4.3 uv pip install --no-build-isolation --no-deps \
    git+https://github.com/sagearc/megablocks-rocm.git@rocm-histogram-hipify-fix
```

## 5. Install the OLMo project

```bash
uv pip install -e ".[train]"
```

## 6. Set `moe_dropless: false` in your configs

This is the critical step that distinguishes the legacy path. Dropless dMoE
crashes the GPU on triton-rocm 3.5.1, so all configs must be flipped:

```bash
sed -i 's/moe_dropless: true/moe_dropless: false/' configs/olmoe/ablations/*.yaml
```

(Same warning: don't commit — leave the repo default at `moe_dropless: true`
for the main 2.11.0 path.)

## 7. Smoke test (single GPU)

```bash
mkdir -p runs slurm_output
ROCM_HOME=/opt/rocm-6.4.3 HIP_VISIBLE_DEVICES=5 \
  .venv/bin/torchrun --nproc_per_node=1 --master_port=29506 \
  scripts/train.py configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml
```

## 8. 2-GPU FSDP test

```bash
ROCM_HOME=/opt/rocm-6.4.3 HIP_VISIBLE_DEVICES=1,5 \
  .venv/bin/torchrun --nproc_per_node=2 --master_port=29507 \
  scripts/train.py configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml
```

---

## What's different from the main 2.11.0 path

| | main (`README_ROCM.md`) | legacy (this file) |
|--|--|--|
| torch | `2.11.0+rocm7.2` | `2.9.1+rocm6.4` |
| triton-rocm | `3.6.0` | `3.5.1` |
| Wheel index | `download.pytorch.org/whl/rocm7.2` | `download.pytorch.org/whl/rocm6.4` |
| ROCm runtime | bundled in wheel (7.2) | bundled in wheel (6.4) |
| Host hipcc | `/opt/rocm-6.4.3` | `/opt/rocm-6.4.3` |
| `moe_dropless` | `true` (dMoE, dropless) | `false` (MoE, capacity-based) |
| Training dynamics | matches CUDA path | drops ~1–5% of overflow tokens |
| Throughput (smoketest, 1 GPU MI325X) | ~24k tok/s | ~32k tok/s |

The 2.9.1 path is *faster* per-token because capacity-based MoE skips overflow
tokens, so there's literally less work per forward pass. But it's not
equivalent training: tokens that should have gone to a busy expert are
silently dropped. For research that aims to reproduce CUDA dropless training,
use the main path.

---

## Known limitations of this path

1. **No dropless MoE** — the central reason this is the fallback. `stk`'s
   triton sparse kernels crash on triton-rocm 3.5.1 with `Memory access fault
   by GPU node-X (Agent handle: 0x...) on address 0x... Reason: Unknown.`
   Confirmed across multiple rebuilds and venv resets.
2. Same `flash_attn_varlen` and ABI caveats as the main path — see
   `README_ROCM.md` for details.

## When to use this path

- You can't install or test `torch==2.11.0+rocm7.2` for some external reason
- You're reproducing an older experiment that ran on this exact stack
- You hit an ABI issue with the rocm7.2 wheel on your host kernel and need
  to validate against the matching-version stack

For new training runs, use the main `README_ROCM.md` path.
