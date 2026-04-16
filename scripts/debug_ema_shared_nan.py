"""Instrumented forward to pin down where ema+shared produces NaN at full scale.

Builds the same 1.2B MoE as the failing config, runs a single forward, and
asserts non-NaN at each layer output and inside the EMA hook.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from olmo.config import ModelConfig, MoERoutingType
from olmo.model import OLMo

torch.manual_seed(6198)

cfg = ModelConfig(
    d_model=2048,
    n_heads=16,
    n_layers=8,
    mlp_ratio=2,
    weight_tying=False,
    alibi=False,
    rope=True,
    flash_attention=True,
    attention_layer_norm=True,
    include_bias=False,
    block_type="moe",
    layer_norm_type="rms",
    layer_norm_with_affine=True,
    bias_for_layer_norm=False,
    attention_layer_norm_with_affine=False,
    activation_type="swiglu",
    max_sequence_length=4096,
    vocab_size=50280,
    embedding_size=50304,
    eos_token_id=0,
    pad_token_id=1,
    init_device="cuda",
    init_fn="normal",
    init_cutoff_factor=3,
    moe_top_k=2,
    moe_num_experts=8,
    moe_dropless=True,
    moe_mlp_impl="sparse",
    moe_zloss_weight=0.0,
    moe_loss_weight=0.0,
    moe_router_ema_normalize=True,
    moe_shared_expert=True,
    precision="amp_bf16",
)

print("Building model...")
model = OLMo(cfg).cuda().train()
model.reset_parameters()

# Hook every block output to find where NaN first appears
def make_probe(name):
    def probe(module, input, output):
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        if torch.isnan(t).any() or torch.isinf(t).any():
            nc = torch.isnan(t).sum().item()
            ic = torch.isinf(t).sum().item()
            print(f"!!! {name}: NaN={nc} Inf={ic} shape={tuple(t.shape)} max={t.abs().max().item():.3e}")
        else:
            print(f"    {name}: max_abs={t.abs().max().item():.3e}")
    return probe

for i, blk in enumerate(model.transformer.blocks):
    blk.register_forward_hook(make_probe(f"block[{i}]"))
    blk.ffn.register_forward_hook(make_probe(f"block[{i}].ffn"))
    if blk.ffn.shared_expert is not None:
        blk.ffn.shared_expert.register_forward_hook(make_probe(f"block[{i}].ffn.shared_expert"))
    blk.ffn.experts.register_forward_hook(make_probe(f"block[{i}].ffn.experts"))

# Small input
B, T = 2, 4096
x = torch.randint(0, 50280, (B, T), device="cuda")

print("Forward pass...")
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    out = model(x)
logits = out.logits
print(f"final logits: shape={tuple(logits.shape)} "
      f"max_abs={logits.abs().max().item():.3e} "
      f"NaN={torch.isnan(logits).sum().item()} "
      f"Inf={torch.isinf(logits).sum().item()}")

# Compute cross-entropy manually
labels = torch.randint(0, 50280, (B, T), device="cuda")
ce = torch.nn.functional.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1))
print(f"ce loss: {ce.item()}")
