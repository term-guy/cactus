"""
Apple CoreML model conversion for cactus NPU acceleration.

Produces .mlpackage files consumed by the cactus C++ ANE backend.
Each component has a specific input/output interface defined in npu.h / npu_ane.mm.

Supported conversions:
  gemma4-vision    → vision_encoder.mlpackage
  gemma4-audio     → audio_encoder.mlpackage
  gemma4-prefill   → model.mlpackage   (LLM prefill stack for NPU-accelerated chunked prefill)
  whisper-encoder  → model.mlpackage

model_id_or_path can be a HuggingFace model ID ("openai/whisper-tiny") or a local
directory containing HuggingFace model files (config.json + safetensors/bin weights).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ── RoPE and norm helpers ──────────────────────────────────────────────────────

def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return (x.float() / norm * weight).to(x.dtype)


def _rms_norm_ones(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization without a learned scale — matches the all-ones v_norm in Gemma4 vision."""
    norm = (x.float().pow(2).mean(-1, keepdim=True) + eps).sqrt()
    return (x.float() / norm).to(x.dtype)


def _apply_2d_rope(
    x: torch.Tensor,     # [N, num_heads, head_dim]
    cos: torch.Tensor,   # [N, 1, head_dim]
    sin: torch.Tensor,   # [N, 1, head_dim]
) -> torch.Tensor:
    """
    2-D rotary position encoding matching compute_2d_rope_tables in
    model_gemma4_vision.cpp.

    head_dim is split into [x_half | y_half].  Each half is rotated
    using its own position frequencies via the [-b, a] rotation:
        rotated = x * cos + [-x[q:], x[:q]] * sin
    where q = len(half) // 2.
    """
    half = x.shape[-1] // 2
    quarter = half // 2

    first = x[..., :half]
    second = x[..., half:]

    cos_x, sin_x = cos[..., :half], sin[..., :half]
    a, b = first[..., :quarter], first[..., quarter:]
    rot_first = first * cos_x + torch.cat([-b, a], dim=-1) * sin_x

    cos_y, sin_y = cos[..., half:], sin[..., half:]
    c, d = second[..., :quarter], second[..., quarter:]
    rot_second = second * cos_y + torch.cat([-d, c], dim=-1) * sin_y

    return torch.cat([rot_first, rot_second], dim=-1)


def _build_causal_mask(
    seq_len: int,
    window: int = 0,
    dtype=torch.float16,
) -> torch.Tensor:
    """
    Additive causal attention mask, optionally sliding-window restricted.
    Returns [1, seq_len, seq_len]: 0 where attention is allowed, -inf elsewhere.
    window=0 means unrestricted causal (global attention).
    """
    i = torch.arange(seq_len).unsqueeze(1)  # [S, 1]
    j = torch.arange(seq_len).unsqueeze(0)  # [1, S]
    allowed = j <= i
    if window > 0:
        allowed = allowed & ((i - j) < window)
    return torch.where(
        allowed,
        torch.zeros(seq_len, seq_len),
        torch.full((seq_len, seq_len), float('-inf')),
    ).unsqueeze(0).to(dtype)  # [1, S, S]


def _apply_rope_1d(
    x: torch.Tensor,    # [S, H, D]
    cos: torch.Tensor,  # [S, 1, rot_half]   rot_half <= D//2
    sin: torch.Tensor,  # [S, 1, rot_half]
) -> torch.Tensor:
    """
    Split-half 1-D RoPE matching apply_partial_rope in model_gemma4.cpp.
    Rotates the first 2*rot_half dimensions; remaining dims pass through unchanged.
    """
    rot_half = cos.shape[-1]
    half = x.shape[-1] // 2
    x1r = x[..., :rot_half]
    x2r = x[..., half : half + rot_half]
    new_x1r = x1r * cos - x2r * sin
    new_x2r = x2r * cos + x1r * sin
    if rot_half < half:
        return torch.cat([new_x1r, x[..., rot_half:half], new_x2r, x[..., half + rot_half:]], dim=-1)
    return torch.cat([new_x1r, new_x2r], dim=-1)


# ── Vision encoder ─────────────────────────────────────────────────────────────

class _VisionEncoderLayer(nn.Module):
    """
    Single Gemma4 SigLIP2 vision encoder layer.

    Architecture (from build_vision_transformer_block in model_gemma4_vision.cpp):
      normed      = rms_norm(hidden, input_layernorm)
      attn_raw    = attention(Q, K, V, mask)           # 2-D RoPE on Q/K
      attn        = rms_norm(attn_raw, post_attention_layernorm)
      residual    = hidden + attn
      pre_mlp     = rms_norm(residual, pre_feedforward_layernorm)
      mlp_raw     = gelu(gate_proj) * up_proj → down_proj
      mlp         = rms_norm(mlp_raw, post_feedforward_layernorm)
      out         = residual + mlp
      out        *= layer_scalar  (optional)
    """

    def __init__(self, hf_layer: nn.Module, num_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.scale = head_dim ** -0.5

        attn = hf_layer.self_attn
        self.q_proj = attn.q_proj
        self.k_proj = attn.k_proj
        self.v_proj = attn.v_proj
        self.o_proj = attn.out_proj
        self.q_norm_w = nn.Parameter(attn.q_norm.weight.data.clone())
        self.k_norm_w = nn.Parameter(attn.k_norm.weight.data.clone())

        self.input_norm = hf_layer.input_layernorm
        self.post_attn_norm = hf_layer.post_attention_layernorm
        self.pre_ffn_norm = hf_layer.pre_feedforward_layernorm
        self.post_ffn_norm = hf_layer.post_feedforward_layernorm

        # MLP: gate_proj / up_proj / down_proj
        mlp = hf_layer.mlp
        self.gate_proj = mlp.gate_proj
        self.up_proj = mlp.up_proj
        self.down_proj = mlp.down_proj

        # Optional per-layer scale (multiplied onto the block output)
        scalar_src = getattr(hf_layer, 'layer_scale', None) or getattr(hf_layer, 'layer_scalar', None)
        if scalar_src is not None:
            data = scalar_src.data if isinstance(scalar_src, torch.Tensor) else scalar_src.weight.data
            self.layer_scalar = nn.Parameter(data.clone())
            self.has_scalar = True
        else:
            self.has_scalar = False

    def forward(
        self,
        h: torch.Tensor,     # [N, embed_dim]
        cos: torch.Tensor,   # [N, 1, head_dim]
        sin: torch.Tensor,   # [N, 1, head_dim]
        mask: torch.Tensor,  # [1, N, N]  additive attention mask
    ) -> torch.Tensor:
        N, D = h.shape

        # ── Attention block ──────────────────────────────────────────────────
        residual = h
        x = self.input_norm(h)

        q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        q = _rms_norm(q, self.q_norm_w, self.eps)
        k = _rms_norm(k, self.k_norm_w, self.eps)
        v = _rms_norm_ones(v, self.eps)

        q = _apply_2d_rope(q, cos, sin)
        k = _apply_2d_rope(k, cos, sin)

        # [num_heads, N, head_dim]
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [nh, N, N]
        scores = scores + mask                                         # broadcast [1, N, N]
        attn = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        out = torch.matmul(attn, v)                                   # [nh, N, hd]

        out = out.transpose(0, 1).reshape(N, D)
        out = self.o_proj(out)
        out = self.post_attn_norm(out)
        h = residual + out

        # ── MLP block ────────────────────────────────────────────────────────
        residual = h
        x = self.pre_ffn_norm(h)
        gate = torch.nn.functional.gelu(self.gate_proj(x))
        up = self.up_proj(x)
        mlp_out = self.down_proj(gate * up)
        mlp_out = self.post_ffn_norm(mlp_out)
        h = residual + mlp_out

        if self.has_scalar:
            h = h * self.layer_scalar

        return h


class Gemma4VisionEncoderCoreML(nn.Module):
    """
    Gemma4 SigLIP2 encoder stack, CoreML-traceable.

    Cactus C++ runtime interface  (vision_encoder.mlpackage):
      hidden_states  [N, embed_dim]    fp16 — patch embeddings computed CPU-side
      cos_full       [N, 1, head_dim]  fp16 — 2-D RoPE cosine table
      sin_signed     [N, 1, head_dim]  fp16 — 2-D RoPE sine table
      attention_mask [1, N, N]         fp16 — additive padding mask (0 or −65504)
      → output       [N, embed_dim]    fp16
    """

    def __init__(
        self,
        hf_vision_tower: nn.Module,
        num_heads: int,
        head_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            _VisionEncoderLayer(layer, num_heads, head_dim, eps)
            for layer in hf_vision_tower.encoder.layers
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos_full: torch.Tensor,
        sin_signed: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = hidden_states
        for layer in self.layers:
            h = layer(h, cos_full, sin_signed, attention_mask)
        return h


# ── Audio encoder ──────────────────────────────────────────────────────────────

class Gemma4AudioEncoderCoreML(nn.Module):
    """
    Gemma4 Conformer audio encoder, CoreML-traceable.

    Cactus C++ runtime interface  (audio_encoder.mlpackage):
      x        [1, 1, T, mel_bins]  fp16 — mel spectrogram (zero-padded to max_frames)
      → output [T', hidden_dim]     fp16 — encoded audio features

    T' = ((T + 1) // 2 + 1) // 2  after the two conv2 downsampling layers.
    """

    def __init__(self, hf_audio_tower: nn.Module):
        super().__init__()
        self.audio_tower = hf_audio_tower

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.audio_tower(x)
        # HF models can return a ModelOutput object; unwrap to a plain tensor.
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)):
            return out[0]
        last = getattr(out, 'last_hidden_state', None)
        if last is not None:
            return last
        return out[0]


# ── CoreML conversion helpers ──────────────────────────────────────────────────

def _require_coremltools():
    try:
        import coremltools as ct
        return ct
    except ImportError:
        raise ImportError(
            "coremltools is required for Apple CoreML conversion.\n"
            "Install with:  pip install coremltools"
        )


def _get_submodule(model: nn.Module, *attr_path: str) -> Optional[nn.Module]:
    """Walk a dotted attribute path; return None if any step is missing."""
    obj = model
    for attr in attr_path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


def _resolve_vision_tower(model: nn.Module) -> nn.Module:
    """Find the Gemma4 vision tower regardless of nesting depth."""
    # Standard HF: model.model.vision_tower
    vt = _get_submodule(model, 'model', 'vision_tower')
    if vt is not None:
        return vt
    # Flat structure
    vt = _get_submodule(model, 'vision_tower')
    if vt is not None:
        return vt
    raise RuntimeError(
        "Cannot find vision_tower in model.  "
        "Expected model.model.vision_tower or model.vision_tower."
    )


def _resolve_audio_tower(model: nn.Module) -> nn.Module:
    at = _get_submodule(model, 'model', 'audio_tower')
    if at is not None:
        return at
    at = _get_submodule(model, 'audio_tower')
    if at is not None:
        return at
    raise RuntimeError(
        "Cannot find audio_tower in model.  "
        "Expected model.model.audio_tower or model.audio_tower."
    )


def _load_gemma4(
    model_id: str,
    token: Optional[str],
    cache_dir: Optional[str],
) -> nn.Module:
    from transformers import AutoModelForCausalLM
    import logging as _log
    _log.getLogger("transformers").setLevel(_log.ERROR)

    logger.info(f"Loading {model_id} (fp16)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=token,
        cache_dir=cache_dir,
    )
    model.eval()
    return model


def _vision_dims(hf_cfg) -> dict:
    vc = getattr(hf_cfg, 'vision_config', hf_cfg)
    num_heads = getattr(vc, 'num_attention_heads', 16)
    hidden = getattr(vc, 'hidden_size', 1152)
    head_dim = getattr(vc, 'head_dim', hidden // num_heads)
    eps = getattr(vc, 'layer_norm_eps', 1e-6)
    default_output = int(getattr(vc, 'default_output_length', 256))
    pool_k = int(getattr(vc, 'pooling_kernel_size', 2))
    return {
        'num_heads': num_heads,
        'head_dim': head_dim,
        'embed_dim': hidden,
        'eps': eps,
        'max_patches': default_output * pool_k * pool_k,
    }


def _audio_dims(hf_cfg) -> dict:
    ac = getattr(hf_cfg, 'audio_config', hf_cfg)
    mel_bins = int(getattr(ac, 'input_feat_size', 128))
    # 30 s at 100 mel frames/s → 3000; Gemma4 uses shorter windows in practice
    max_frames = int(getattr(ac, 'max_source_positions', getattr(ac, 'max_length', 3000)))
    return {'mel_bins': mel_bins, 'max_frames': max_frames}


def _save_mlpackage(
    module: nn.Module,
    example_inputs: tuple,
    input_names: list[str],
    output_name: str,
    out_path: Path,
) -> None:
    import numpy as np
    ct = _require_coremltools()

    logger.info("Tracing model with torch.jit.trace...")
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs)

    ct_inputs = [
        ct.TensorType(name=name, shape=list(inp.shape), dtype=np.float16)
        for name, inp in zip(input_names, example_inputs)
    ]
    ct_outputs = [ct.TensorType(name=output_name, dtype=np.float16)]

    logger.info("Converting to CoreML (fp16)...")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    logger.info(f"Saved CoreML model → {out_path}")


def _save_mlpackage_multi(
    module: nn.Module,
    example_inputs: tuple,
    input_names: list,
    output_names: list,
    out_path: Path,
) -> None:
    """Like _save_mlpackage but for modules that return a tuple of tensors."""
    import numpy as np
    ct = _require_coremltools()

    logger.info("Tracing model with torch.jit.trace...")
    module.eval()
    with torch.no_grad():
        traced = torch.jit.trace(module, example_inputs)

    ct_inputs = [
        ct.TensorType(name=name, shape=list(inp.shape), dtype=np.float16)
        for name, inp in zip(input_names, example_inputs)
    ]
    ct_outputs = [ct.TensorType(name=name, dtype=np.float16) for name in output_names]

    logger.info(f"Converting to CoreML (fp16, {len(output_names)} outputs)...")
    mlmodel = ct.convert(
        traced,
        inputs=ct_inputs,
        outputs=ct_outputs,
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,
    )

    if out_path.exists():
        shutil.rmtree(out_path)
    mlmodel.save(str(out_path))
    logger.info(f"Saved CoreML model → {out_path}")


# ── Public conversion functions ────────────────────────────────────────────────

def convert_gemma4_vision_encoder(
    model_id: str,
    output_dir: Path,
    bits: str = "4",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Convert the Gemma4 SigLIP2 vision encoder to CoreML.

    Produces:  <output_dir>/vision_encoder.mlpackage

    The output is bundled into the apple weights zip by publish_to_hf.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_gemma4(model_id, token, cache_dir)
    vision_tower = _resolve_vision_tower(model)
    dims = _vision_dims(model.config)

    logger.info(
        f"Building vision encoder wrapper  "
        f"(layers={len(vision_tower.encoder.layers)}, "
        f"heads={dims['num_heads']}, head_dim={dims['head_dim']}, "
        f"max_patches={dims['max_patches']})"
    )
    wrapper = Gemma4VisionEncoderCoreML(
        vision_tower,
        num_heads=dims['num_heads'],
        head_dim=dims['head_dim'],
        eps=dims['eps'],
    ).eval()

    N = dims['max_patches']
    D = dims['embed_dim']
    H = dims['head_dim']

    example = (
        torch.zeros(N, D, dtype=torch.float16),         # hidden_states
        torch.ones(N, 1, H, dtype=torch.float16),        # cos_full
        torch.zeros(N, 1, H, dtype=torch.float16),       # sin_signed
        torch.zeros(1, N, N, dtype=torch.float16),       # attention_mask
    )

    out_path = output_dir / "vision_encoder.mlpackage"
    _save_mlpackage(
        wrapper,
        example,
        ["hidden_states", "cos_full", "sin_signed", "attention_mask"],
        "output",
        out_path,
    )
    return out_path


def convert_gemma4_audio_encoder(
    model_id: str,
    output_dir: Path,
    bits: str = "4",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Convert the Gemma4 Conformer audio encoder to CoreML.

    Produces:  <output_dir>/audio_encoder.mlpackage
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_gemma4(model_id, token, cache_dir)
    audio_tower = _resolve_audio_tower(model)
    dims = _audio_dims(model.config)

    logger.info(
        f"Building audio encoder wrapper  "
        f"(max_frames={dims['max_frames']}, mel_bins={dims['mel_bins']})"
    )
    wrapper = Gemma4AudioEncoderCoreML(audio_tower).eval()

    T = dims['max_frames']
    F = dims['mel_bins']
    example = (torch.zeros(1, 1, T, F, dtype=torch.float16),)

    out_path = output_dir / "audio_encoder.mlpackage"
    _save_mlpackage(wrapper, example, ["x"], "output", out_path)
    return out_path


# ── Gemma4 LLM prefill (model.mlpackage) ──────────────────────────────────────

_PREFILL_CHUNK_SIZE = 512


def _text_dims(hf_cfg) -> dict:
    """Extract Gemma4 language-model dimensions from a HF config object."""
    tc = getattr(hf_cfg, 'text_config', hf_cfg)
    hidden   = getattr(tc, 'hidden_size',                   2048)
    n_heads  = getattr(tc, 'num_attention_heads',           8)
    n_kv     = getattr(tc, 'num_key_value_heads',           n_heads)
    h_dim    = getattr(tc, 'head_dim',                      hidden // n_heads)
    eps      = getattr(tc, 'rms_norm_eps',                  1e-6)
    window   = getattr(tc, 'sliding_window',                1024)
    g_h_dim  = getattr(tc, 'global_head_dim',               0) or (h_dim * 2)
    g_n_kv   = getattr(tc, 'num_global_key_value_heads',    n_kv)
    g_rot    = float(getattr(tc, 'global_partial_rotary_factor', 1.0))
    rope_l   = float(getattr(tc, 'rope_local_base_freq',
                     getattr(tc, 'rope_theta', 10000.0)))
    rope_g   = float(getattr(tc, 'rope_global_base_freq',
                     getattr(tc, 'rope_theta', 10000.0)))
    return dict(
        hidden_size=hidden,
        num_heads=n_heads,
        num_kv_heads=n_kv,
        head_dim=h_dim,
        global_head_dim=g_h_dim,
        num_global_kv_heads=g_n_kv,
        global_partial_rotary=g_rot,
        eps=eps,
        sliding_window=window,
        rope_local=rope_l,
        rope_global=rope_g,
        layer_types=getattr(tc, 'layer_types', None),
    )


def _resolve_language_model(model: nn.Module) -> nn.Module:
    """Return the core language model (has .layers and .norm) from a Gemma4 wrapper."""
    for attrs in [
        ('language_model', 'model'),
        ('model', 'language_model', 'model'),
        ('language_model',),
        ('model',),
    ]:
        obj = model
        for a in attrs:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, 'layers'):
            return obj
    raise RuntimeError(
        "Cannot find transformer layers in model. "
        "Expected model.language_model.model.layers or similar."
    )


def _layer_is_global(hf_layer: nn.Module, layer_types: Optional[list], idx: int) -> bool:
    """Return True when layer idx uses global (full) attention."""
    if layer_types is not None and idx < len(layer_types):
        t = layer_types[idx].lower()
        return 'global' in t or 'full' in t
    sw = getattr(getattr(hf_layer, 'self_attn', None), 'sliding_window', None)
    return sw is None or sw == 0


class _Gemma4PrefillLayer(nn.Module):
    """
    Single Gemma4 text transformer layer, CoreML-traceable.

    Mirrors build_transformer_block + build_attention in model_gemma4.cpp.
    Per-Layer Inputs (PLI) are omitted — they require token IDs that the NPU
    prefill interface does not pass (only embeddings and position offset are
    available).

    Outputs (h, k_cache, v_cache):
      h:       [S, hidden_dim]         updated hidden states
      k_cache: [S, kv_heads, head_dim] post-RoPE keys  (matches C++ cache layout)
      v_cache: [S, kv_heads, head_dim] RMS-normed values (no RoPE, matches C++)
    """

    def __init__(
        self,
        hf_layer: nn.Module,
        *,
        num_heads: int,
        num_kv_heads: int,
        eps: float,
        attn_mask: torch.Tensor,  # [1, S, S]
    ):
        super().__init__()

        attn = hf_layer.self_attn
        # head_dim inferred from the per-head q_norm weight shape
        self.head_dim  = attn.q_norm.weight.shape[0]
        self.num_heads = num_heads
        self.kv_heads  = num_kv_heads
        # Gemma4 uses attention_scale = 1.0 (see engine_model.cpp:128)
        self.attn_scale = 1.0
        self.eps = eps
        self.n_rep = num_heads // num_kv_heads

        self.q_proj  = attn.q_proj
        self.k_proj  = attn.k_proj
        self.v_proj  = attn.v_proj
        self.o_proj  = attn.o_proj
        self.q_norm_w = nn.Parameter(attn.q_norm.weight.data.clone())
        self.k_norm_w = nn.Parameter(attn.k_norm.weight.data.clone())
        # V uses all-ones RMS norm (no learned scale) — matches v_norm_ones in C++
        self.register_buffer('v_norm_ones', torch.ones(self.head_dim, dtype=torch.float16))

        self.input_norm     = hf_layer.input_layernorm
        self.post_attn_norm = hf_layer.post_attention_layernorm
        self.pre_ffn_norm   = hf_layer.pre_feedforward_layernorm
        self.post_ffn_norm  = hf_layer.post_feedforward_layernorm

        mlp = getattr(hf_layer, 'mlp', None)
        if mlp is None:
            raise ValueError(
                "Layer has no 'mlp' attribute — MoE layers are not yet supported "
                "in Gemma4 NPU prefill conversion."
            )
        self.gate_proj = mlp.gate_proj
        self.up_proj   = mlp.up_proj
        self.down_proj  = mlp.down_proj

        # Causal (+ optional sliding-window) mask — constant in the traced graph
        self.register_buffer('attn_mask', attn_mask)  # [1, S, S]

    def forward(
        self,
        h: torch.Tensor,    # [S, hidden_dim]
        cos: torch.Tensor,  # [S, 1, rot_half]
        sin: torch.Tensor,  # [S, 1, rot_half]
    ):
        S = h.shape[0]
        attn_dim = self.num_heads * self.head_dim

        # ── Attention ────────────────────────────────────────────────────────
        residual = h
        x = self.input_norm(h)

        q = self.q_proj(x).view(S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(S, self.kv_heads,  self.head_dim)
        v = self.v_proj(x).view(S, self.kv_heads,  self.head_dim)

        q = _rms_norm(q, self.q_norm_w, self.eps)
        k = _rms_norm(k, self.k_norm_w, self.eps)
        v = _rms_norm(v, self.v_norm_ones, self.eps)   # all-ones norm

        q     = _apply_rope_1d(q, cos, sin)
        k_rot = _apply_rope_1d(k, cos, sin)

        # KV cache: K after RoPE, V after norm only — matches C++ cache layout
        k_cache = k_rot  # [S, kv_heads, head_dim]
        v_cache = v       # [S, kv_heads, head_dim]

        # GQA: expand KV heads to match Q heads
        if self.n_rep > 1:
            k_exp = k_rot.repeat_interleave(self.n_rep, dim=1)
            v_exp = v.repeat_interleave(self.n_rep, dim=1)
        else:
            k_exp = k_rot
            v_exp = v

        qt = q.transpose(0, 1)     # [H, S, hd]
        kt = k_exp.transpose(0, 1)
        vt = v_exp.transpose(0, 1)

        scores  = torch.matmul(qt, kt.transpose(-2, -1)) * self.attn_scale
        scores  = scores + self.attn_mask                  # broadcast [1, S, S]
        weights = torch.softmax(scores.float(), dim=-1).to(h.dtype)
        out     = torch.matmul(weights, vt)                # [H, S, hd]

        out = out.transpose(0, 1).reshape(S, attn_dim)
        out = self.post_attn_norm(self.o_proj(out))
        h   = residual + out

        # ── MLP ──────────────────────────────────────────────────────────────
        residual = h
        x    = self.pre_ffn_norm(h)
        gate = torch.nn.functional.gelu(self.gate_proj(x))
        up   = self.up_proj(x)
        h    = residual + self.post_ffn_norm(self.down_proj(gate * up))

        return h, k_cache, v_cache


class Gemma4PrefillCoreML(nn.Module):
    """
    Gemma4 LLM prefill stack, CoreML-traceable.

    Cactus C++ runtime interface (model.mlpackage):
      x      [chunk_size, hidden_dim]  fp16 — pre-scaled token embeddings
      offset [1]                       fp16 — position index of the first token

    Returns a flat tuple (hidden, k_0, v_0, k_1, v_1, …, k_{N-1}, v_{N-1}):
      hidden [chunk_size, hidden_dim]
      k_i    [chunk_size, kv_heads_i, head_dim_i]  — post-RoPE keys
      v_i    [chunk_size, kv_heads_i, head_dim_i]  — RMS-normed values

    The C++ ANEPrefill runtime infers chunk_size, hidden_dim, num_layers,
    num_kv_heads, and head_dim from the model description, then populates the
    KV cache layer by layer via update_from_npu.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        dims: dict,
        chunk_size: int = _PREFILL_CHUNK_SIZE,
    ):
        super().__init__()

        head_dim   = dims['head_dim']
        g_h_dim    = dims['global_head_dim']
        rope_l     = dims['rope_local']
        rope_g     = dims['rope_global']
        g_rot      = dims['global_partial_rotary']
        eps        = dims['eps']
        num_heads  = dims['num_heads']
        n_kv       = dims['num_kv_heads']
        g_n_kv     = dims['num_global_kv_heads']
        window     = dims['sliding_window']
        layer_types = dims['layer_types']

        # Local RoPE: full rotation over head_dim
        self.register_buffer(
            'inv_freq_local',
            1.0 / (rope_l ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)),
        )  # [head_dim // 2]

        # Global RoPE: partial rotation (rot_dim = global_head_dim * g_rot, rounded to even)
        rot_dim_g = max(2, int(g_h_dim * g_rot) & ~1)
        self.register_buffer(
            'inv_freq_global',
            1.0 / (rope_g ** (torch.arange(0, rot_dim_g, 2, dtype=torch.float32) / g_h_dim)),
        )  # [rot_dim_g // 2]

        # Pre-compute causal masks per attention type
        local_mask  = _build_causal_mask(chunk_size, window=window)
        global_mask = _build_causal_mask(chunk_size, window=0)

        # Build per-layer wrappers
        self._is_global: List[bool] = []
        layers_list: List[nn.Module] = []
        for i, hf_layer in enumerate(lm_model.layers):
            is_g = _layer_is_global(hf_layer, layer_types, i)
            self._is_global.append(is_g)
            layers_list.append(
                _Gemma4PrefillLayer(
                    hf_layer,
                    num_heads=num_heads,
                    num_kv_heads=g_n_kv if is_g else n_kv,
                    eps=eps,
                    attn_mask=global_mask if is_g else local_mask,
                )
            )
        self.layers = nn.ModuleList(layers_list)
        self.output_norm = lm_model.norm
        self.chunk_size = chunk_size

    def forward(
        self,
        x: torch.Tensor,       # [S, hidden_dim]
        offset: torch.Tensor,  # [1]  fp16
    ) -> tuple:
        S = x.shape[0]

        # Absolute positions for this chunk
        pos = (
            torch.arange(S, dtype=torch.float32)
            + offset.float().squeeze()
        )  # [S]

        # Local RoPE tables: [S, 1, head_dim//2]
        freqs_l = pos.unsqueeze(1) * self.inv_freq_local.float().unsqueeze(0)
        cos_l   = torch.cos(freqs_l).unsqueeze(1).to(x.dtype)
        sin_l   = torch.sin(freqs_l).unsqueeze(1).to(x.dtype)

        # Global RoPE tables: [S, 1, rot_dim_g//2]
        freqs_g = pos.unsqueeze(1) * self.inv_freq_global.float().unsqueeze(0)
        cos_g   = torch.cos(freqs_g).unsqueeze(1).to(x.dtype)
        sin_g   = torch.sin(freqs_g).unsqueeze(1).to(x.dtype)

        h = x
        kv_list = []
        for layer, is_g in zip(self.layers, self._is_global):
            cos = cos_g if is_g else cos_l
            sin = sin_g if is_g else sin_l
            h, k, v = layer(h, cos, sin)
            kv_list.append(k)
            kv_list.append(v)

        h = self.output_norm(h)

        # Flat tuple: (hidden, k_0, v_0, k_1, v_1, …)
        return tuple([h] + kv_list)


def convert_gemma4_prefill(
    model_id: str,
    output_dir: Path,
    chunk_size: int = _PREFILL_CHUNK_SIZE,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Convert the Gemma4 text LLM prefill stack to CoreML.

    Produces: <output_dir>/model.mlpackage

    The model takes pre-scaled token embeddings and a position offset, then
    returns the hidden states and per-layer KV cache pairs.  The cactus C++
    runtime uses this for NPU-accelerated chunked prefill (prefill_npu).

    chunk_size sets the fixed sequence length the CoreML model processes per
    call; it becomes the leading dimension of the 'x' input and each k_i/v_i
    output.  The runtime reads this from the model description automatically.

    Note: Per-Layer Inputs (PLI) are not included in the NPU model because the
    prefill interface (prefill_chunk_direct) only provides token embeddings and
    a position offset — the token IDs needed for the PLI embedding lookup are
    not available at that point.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = _load_gemma4(model_id, token, cache_dir)
    lm    = _resolve_language_model(model)
    dims  = _text_dims(model.config)

    n_layers = len(lm.layers)
    logger.info(
        f"Building Gemma4 prefill wrapper  "
        f"(layers={n_layers}, chunk_size={chunk_size}, "
        f"hidden={dims['hidden_size']}, heads={dims['num_heads']}, "
        f"head_dim={dims['head_dim']})"
    )

    wrapper = Gemma4PrefillCoreML(lm, dims, chunk_size=chunk_size).eval()

    S = chunk_size
    H = dims['hidden_size']
    example = (
        torch.zeros(S, H, dtype=torch.float16),  # x
        torch.zeros(1,    dtype=torch.float16),   # offset
    )

    # Output names: hidden then k_i / v_i per layer
    output_names = ['hidden']
    for i in range(n_layers):
        output_names += [f'k_{i}', f'v_{i}']

    out_path = output_dir / 'model.mlpackage'
    _save_mlpackage_multi(wrapper, example, ['x', 'offset'], output_names, out_path)
    return out_path


# ── Whisper encoder ────────────────────────────────────────────────────────────

class WhisperEncoderCoreML(nn.Module):
    """
    Wraps HF WhisperEncoder for CoreML tracing.

    Cactus C++ runtime interface  (model.mlpackage):
      x        [1, num_mel_bins, 3000]  fp16 — mel spectrogram
      → output [1, 1500, d_model]       fp16 — encoded audio features
    """

    def __init__(self, hf_encoder: nn.Module):
        super().__init__()
        self.encoder = hf_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        if isinstance(out, torch.Tensor):
            return out
        last = getattr(out, 'last_hidden_state', None)
        if last is not None:
            return last
        if isinstance(out, (tuple, list)):
            return out[0]
        return out


def convert_whisper_encoder(
    model_id_or_path: str,
    output_dir: Path,
    bits: str = "4",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Path:
    """
    Convert the Whisper audio encoder to CoreML.

    Produces:  <output_dir>/model.mlpackage
    """
    from transformers import WhisperModel, AutoConfig

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading Whisper config from {model_id_or_path}...")
    cfg = AutoConfig.from_pretrained(model_id_or_path, token=token, cache_dir=cache_dir)
    num_mel_bins = getattr(cfg, 'num_mel_bins', 80)

    logger.info(f"Loading Whisper model (fp16)...")
    model = WhisperModel.from_pretrained(
        model_id_or_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=token,
        cache_dir=cache_dir,
    )
    model.eval()

    wrapper = WhisperEncoderCoreML(model.encoder).eval()
    example = (torch.zeros(1, num_mel_bins, 3000, dtype=torch.float16),)

    out_path = output_dir / "model.mlpackage"
    _save_mlpackage(wrapper, example, ["x"], "output", out_path)
    return out_path


# ── Auto-detection helpers ─────────────────────────────────────────────────────

def _is_local_path(s: str) -> bool:
    return Path(s).exists()


def _read_cactus_config(path: Path) -> dict:
    """Parse a cactus config.txt (key=value) into a dict."""
    cfg = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            k, _, v = line.partition('=')
            cfg[k.strip()] = v.strip()
    return cfg


def _detect_enc_types(model_id_or_path: str, token: Optional[str] = None) -> List[str]:
    """
    Infer which Apple CoreML encoders to produce for a given model.

    Returns a list of enc_type strings (e.g. ``["gemma4-vision", "gemma4-audio"]``).
    Raises ValueError if the model type is unrecognised or unsupported.
    """
    # Use just the final path component for name-pattern matching
    if _is_local_path(model_id_or_path):
        name = Path(model_id_or_path).name.lower()
    else:
        name = model_id_or_path.lower().split('/')[-1]

    full = model_id_or_path.lower()

    # Fast pattern-based detection (no network/disk I/O)
    if "whisper" in name:
        return ["whisper-encoder"]
    if "moonshine" in name:
        raise ValueError(
            f"Moonshine Apple CoreML conversion is not yet implemented. "
            f"Contributions welcome."
        )
    if "parakeet" in name:
        raise ValueError(
            f"Parakeet Apple CoreML conversion is not yet implemented. "
            f"Contributions welcome."
        )

    is_gemma4_name = any(x in full for x in ("gemma-4", "gemma4", "gemma_4"))

    # For local paths, inspect the directory contents first
    if _is_local_path(model_id_or_path):
        p = Path(model_id_or_path)
        if (p / "config.txt").exists() and not (p / "config.json").exists():
            cactus_cfg = _read_cactus_config(p / "config.txt")
            raise ValueError(
                f"{model_id_or_path!r} is a cactus weights directory (contains config.txt). "
                "Apple CoreML conversion requires the original HuggingFace model weights. "
                "Pass a HuggingFace model ID (e.g. 'openai/whisper-tiny') or a local "
                "directory that contains config.json and the original model files."
            )

    # Load config to detect model_type and available sub-configs
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id_or_path, token=token)
        model_type = getattr(cfg, 'model_type', '').lower()
    except Exception:
        cfg = None
        model_type = ""

    if "whisper" in model_type:
        return ["whisper-encoder"]

    if is_gemma4_name or "gemma4" in model_type or "gemma_4" in model_type:
        enc_types: List[str] = []
        if cfg is not None:
            if getattr(cfg, 'vision_config', None) is not None:
                enc_types.append("gemma4-vision")
            if getattr(cfg, 'audio_config', None) is not None:
                enc_types.append("gemma4-audio")
        enc_types.append("gemma4-prefill")
        return enc_types or ["gemma4-vision", "gemma4-audio", "gemma4-prefill"]

    raise ValueError(
        f"Cannot auto-detect Apple encoder type for {model_id_or_path!r} "
        f"(model_type={model_type!r}). "
        "Specify enc_type explicitly: gemma4-vision, gemma4-audio, gemma4-prefill, whisper-encoder."
    )


# ── Public conversion functions ────────────────────────────────────────────────

def convert_model_for_apple(
    model_id: str,
    enc_type: Optional[str],
    output_dir: Path,
    bits: str = "4",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> Optional[Path]:
    """
    Convert one model component to a CoreML .mlpackage.

    enc_type:
      'gemma4-vision'   → vision_encoder.mlpackage
      'gemma4-audio'    → audio_encoder.mlpackage
      'gemma4-prefill'  → model.mlpackage   (LLM prefill)
      'whisper-encoder' → model.mlpackage
      None              → auto-detect (returns first produced package)

    Returns the path to the produced .mlpackage, or None on failure.
    For all encoders at once use :func:`convert_all_for_apple`.
    """
    if enc_type is None:
        results = convert_all_for_apple(model_id, output_dir, bits, token, cache_dir)
        return results[0] if results else None
    try:
        if enc_type == "gemma4-vision":
            return convert_gemma4_vision_encoder(model_id, output_dir, bits, token, cache_dir)
        if enc_type == "gemma4-audio":
            return convert_gemma4_audio_encoder(model_id, output_dir, bits, token, cache_dir)
        if enc_type == "gemma4-prefill":
            return convert_gemma4_prefill(model_id, output_dir, token=token, cache_dir=cache_dir)
        if enc_type == "whisper-encoder":
            return convert_whisper_encoder(model_id, output_dir, bits, token, cache_dir)
        raise ValueError(
            f"Unknown enc_type {enc_type!r}. "
            "Supported: gemma4-vision, gemma4-audio, gemma4-prefill, whisper-encoder."
        )
    except Exception as exc:
        logger.error(f"Apple conversion failed for {enc_type}: {exc}")
        return None


def convert_all_for_apple(
    model_id_or_path: str,
    output_dir: Path,
    bits: str = "4",
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> List[Path]:
    """
    Auto-detect and convert all applicable Apple CoreML encoders for a model.

    Accepts a HuggingFace model ID or a local directory containing the original
    HuggingFace model files (config.json + weights).

    Returns a list of produced .mlpackage paths (may be empty if conversion fails).
    Raises ValueError for unrecognised or unsupported model types.
    """
    enc_types = _detect_enc_types(model_id_or_path, token)
    logger.info(f"Detected enc_types for {model_id_or_path!r}: {enc_types}")

    results: List[Path] = []
    for enc_type in enc_types:
        path = convert_model_for_apple(model_id_or_path, enc_type, output_dir, bits, token, cache_dir)
        if path is not None:
            results.append(path)
        else:
            logger.warning(f"Skipping {enc_type} — conversion returned None")
    return results
