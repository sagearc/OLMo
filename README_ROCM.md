# OLMoE on AMD ROCm — torch 2.11.0+rocm7.2 (recommended)

Reproduction steps for training OLMoE on an AMD ROCm node. Tested on:
- AMD Instinct MI325X (`gfx942`)
- Host ROCm: 6.4.3 (kernel driver only — torch wheel ships its own ROCm 7.2 user-space)
- Python 3.12, `uv` for env management
- `torch==2.11.0+rocm7.2`, `triton-rocm==3.6.0`

The original `olmo` repo builds for CUDA/H100 and is untouched; this clone lives
alongside it and has the ROCm-specific patches.

> **Why this is the recommended path**: triton-rocm 3.6 (bundled with torch
> 2.11) fixes the GPU memory access fault that earlier ROCm triton hit inside
> `stk`'s sparse kernels, so dropless dMoE works correctly. The 2.9.1 path
> (see `README_ROCM_LEGACY_2.9.1.md`) is forced to fall back to capacity-based
> MoE, which changes training dynamics. Stay on this path unless you have a
> specific reason to downgrade.

## 1. Clone into a new directory

```bash
cd /home/morg/students/sagiahrac/repos
git clone --branch train-olmoe-moe https://github.com/sagearc/olmo.git olmo-rocm
cd olmo-rocm
```

## 2. Create env + install PyTorch ROCm

```bash
uv venv --python 3.12
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/rocm7.2
```

Verify 8 GPUs are visible:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.device_count())"
# expected: 2.11.0+rocm7.2 7.2.... 8
```

> **Note on ROCm versions**: the host has only `/opt/rocm-6.4.3` installed. The
> torch wheel bundles ROCm 7.2 user-space libs (`librocblas.so`, `libhipblaslt.so`,
> `libMIOpen.so`, `librccl.so`, etc.) inside `torch/lib/`, so torch runs against
> ROCm 7.2 internally while the host kernel driver (`amdgpu`) handles the GPU.
> This forward-compatibility works on `gfx942`. C++ extensions (megablocks)
> still build with the host's `/opt/rocm-6.4.3` hipcc.

## 3. Install megablocks from the sagearc fork

The fork is [`sagearc/megablocks-rocm`](https://github.com/sagearc/megablocks-rocm),
branch `rocm-histogram-hipify-fix`, with two commits on top of `ROCm/megablocks`:

1. **`csrc/histogram.h` — hipify workaround for `cub::DeviceHistogram`**
   PyTorch's hipify maps `cub::DeviceScan` / `DeviceRadixSort` / `DeviceSegmentedReduce`
   to their hipcub equivalents but is missing `cub::DeviceHistogram`. Without
   the alias, hipified code fails to compile with `error: use of undeclared
   identifier 'cub'`. The fix is a 4-line `namespace cub { using
   hipcub::DeviceHistogram; }` under `#ifdef __HIP_PLATFORM_AMD__`. See the
   PyTorch PR linked in `pytorch/PR_DESCRIPTION.md`.

2. **`router.py` / `moe.py` — expose logits in router output**
   Port of Muennighoff's `4a25bc7 "Keep logits"` commit. Makes the ROCm fork's
   router return `(scores, logits, weights, indices)` (4-tuple) so OLMo's
   `_router_ema_hook` can use a single unified code path on both CUDA and ROCm.

Install:

```bash
# stanford-stk is a megablocks runtime dep; bypass torch version guard.
uv pip install --no-build-isolation --no-deps stanford-stk==0.7.1

# Build + install megablocks from the sagearc fork branch.
ROCM_HOME=/opt/rocm-6.4.3 uv pip install --no-build-isolation --no-deps \
    git+https://github.com/sagearc/megablocks-rocm.git@rocm-histogram-hipify-fix
```

Quick check:

```bash
.venv/bin/python -c "
from megablocks.layers.moe import MoE
from megablocks.layers.dmoe import dMoE
import megablocks_ops
print('megablocks OK')
"
```

## 4. Install the OLMo project

```bash
uv pip install -e ".[train]"
```

`pyproject.toml` pins `torch==2.11.0` (no local `+rocm` segment — uv ignores it
in version matching), so the existing ROCm install satisfies the constraint
and isn't reinstalled.

## 5. Smoke test (single GPU)

Uses `configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml` — runs 20
steps with dropless dMoE, loss should decrease, checkpoint saved.

```bash
mkdir -p runs slurm_output
ROCM_HOME=/opt/rocm-6.4.3 HIP_VISIBLE_DEVICES=5 \
  .venv/bin/torchrun --nproc_per_node=1 --master_port=29506 \
  scripts/train.py configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml
```

## 6. 2-GPU FSDP test

```bash
ROCM_HOME=/opt/rocm-6.4.3 HIP_VISIBLE_DEVICES=1,5 \
  .venv/bin/torchrun --nproc_per_node=2 --master_port=29507 \
  scripts/train.py configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml
```

## 7. Full training via slurm

```bash
sbatch olmoe-ema-2gpu-rocm.slurm
```

---

## ROCm-specific deltas vs. the CUDA `olmo` repo

Only in `olmo-rocm`; the CUDA repo is untouched.

### `pyproject.toml`
- `requires-python = ">=3.10"` (uv resolver needs ≥3.10 for cached_path)
- `torch==2.11.0` (was `torch>=2.1,<2.5` in CUDA repo)
- No platform-specific `[tool.uv.sources]` routing — torch is installed
  out-of-band from the right wheel index per platform, then the constraint is
  satisfied without reinstall.

### `olmo/config.py`
- Removed `"moe_weight_parallelism": False` from `config_to_moe_args`.
  Upstream `databricks/megablocks` dropped this field in 0.6+; the ROCm fork
  inherits that. OLMo's CUDA path uses `Muennighoff/megablocks@olmoe` (0.5.1)
  which still has it. The field was always set to `False` ("Handled by FSDP"),
  so removal has zero behavioral effect on either path.

### `olmo/model.py` — `_router_ema_hook`
- Unchanged from CUDA path. Both forks now return a 4-tuple
  `(scores, logits, weights, indices)`, so the hook can use a single
  `_, logits, _, _ = output` unpack.

### Configs
- All `configs/olmoe/ablations/*.yaml`:
  - `save_folder` paths `olmo` → `olmo-rocm`.
  - Everything else (`flash_attention: true`, `moe_dropless: true`, MoE
    settings) is unchanged from the CUDA path.
- New `configs/olmoe/ablations/olmoe17-8x1b-rocm-smoketest.yaml` — 20-step
  validation config.

### `flash_attention: true` on ROCm
The `flash_attn` pip package isn't installed on ROCm. `model.py:466-478`
catches `ModuleNotFoundError` and `self.flash_attn_func` stays `None`, so the
attention method falls through to `F.scaled_dot_product_attention`. On torch
2.11+rocm7.2, SDPA dispatches to AOTriton/CK flash kernels automatically.
Equivalent performance to the CUDA flash-attn path.

### New slurm script
- `olmoe-ema-2gpu-rocm.slurm` — sets `ROCM_HOME=/opt/rocm-6.4.3`, points `cd`
  at `olmo-rocm`, uses the 2-GPU config.

---

## Known limitations

1. **Flash-attn varlen / document packing** — the `flash_attn_varlen_func`
   path (`max_doc_len` / `cu_doc_lens` in `OLMoBlock.attention`) asserts that
   flash-attn is available. Not triggered by current configs
   (`generate_doc_lengths: false`), but would need a fallback if enabled.
2. **`hipcc` host/runtime version skew** — megablocks is compiled with
   `/opt/rocm-6.4.3` hipcc but loaded against torch's bundled ROCm 7.2 libs.
   Working so far; if a future torch release breaks ABI compatibility, install
   a matching ROCm toolchain in `$HOME/rocm-7.2` and point `ROCM_HOME` there.

## Fallback path

If torch 2.11.0+rocm7.2 doesn't work for some reason on your environment,
see `README_ROCM_LEGACY_2.9.1.md` for the older `torch==2.9.1+rocm6.4` path.
That path can't use dropless dMoE due to a triton-rocm regression, so it
falls back to capacity-based MoE — different training dynamics.
