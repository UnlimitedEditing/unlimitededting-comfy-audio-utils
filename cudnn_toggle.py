"""UE_CudnnSdpToggle: pass an AUDIO through while switching PyTorch's cuDNN scaled-dot-product-attention backend on or off.

Why: on Graydient runners that install the CUDA 13 torch wheels (torch 2.11+cu130, used by the fast MiniMax H3 workflows),
Stable Audio 3's VAE encoder fails inside torch SDPA with "cuDNN Frontend error: No valid execution plans built"
(seen 2026-11-26, trisulero-sa3-test). Turning the cuDNN SDPA backend off makes PyTorch fall back to the flash / memory-efficient /
math kernels. Put one node (enable = False) in front of the audio model and one (enable = True) after it, so the rest of the
graph keeps cuDNN attention. Pure pass-through: the audio is not modified.
"""
import torch
import torch.nn.functional as F

_ORIG_SDPA = F.scaled_dot_product_attention
_MATH_MAX_SEQ = 8192  # above this (H3 video attention) keep torch's kernel; audio VAEs / DiTs are far below


def _math_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False, **kw):
    """Plain softmax attention. Used for short sequences while the toggle is OFF, because on torch cu130 the
    flag alone did not stop the cuDNN path (2026-11-26: error persisted with 'cuDNN SDPA backend disabled' logged)."""
    if enable_gqa or dropout_p or q.dim() != 4 or max(q.shape[-2], k.shape[-2]) > _MATH_MAX_SEQ:
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale,
                          **({"enable_gqa": enable_gqa} if enable_gqa else {}))
    sc = (q.shape[-1] ** -0.5) if scale is None else scale
    att = torch.matmul(q * sc, k.transpose(-2, -1)).float()
    if is_causal:
        att = att.masked_fill(torch.ones(att.shape[-2:], dtype=torch.bool, device=att.device).triu(1), float("-inf"))
    if attn_mask is not None:
        att = att.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else att + attn_mask.float()
    return torch.matmul(torch.softmax(att, dim=-1).to(v.dtype), v)


class CudnnSdpToggle:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "enable": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    @classmethod
    def IS_CHANGED(cls, audio, enable):
        # the backend flag is process state, never cache this node
        return float("nan")

    def run(self, audio, enable):
        try:
            torch.backends.cuda.enable_cudnn_sdp(bool(enable))
            F.scaled_dot_product_attention = _ORIG_SDPA if enable else _math_sdpa
            print(f"[UE_CudnnSdpToggle] cuDNN SDPA backend {'enabled' if enable else 'disabled'}; "
                  f"short-sequence attention {'= torch default' if enable else 'uses plain softmax (<=%d tokens)' % _MATH_MAX_SEQ}")
        except Exception as exc:  # old torch without the flag: nothing to toggle
            print(f"[UE_CudnnSdpToggle] could not set cuDNN SDPA ({exc}); continuing")
        return (audio,)


NODE_CLASS_MAPPINGS = {"UE_CudnnSdpToggle": CudnnSdpToggle}
NODE_DISPLAY_NAME_MAPPINGS = {"UE_CudnnSdpToggle": "Audio - cuDNN SDPA toggle (pass-through)"}
