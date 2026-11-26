"""UE_CudnnSdpToggle: pass an AUDIO through while switching PyTorch's cuDNN scaled-dot-product-attention backend on or off.

Why: on Graydient runners that install the CUDA 13 torch wheels (torch 2.11+cu130, used by the fast MiniMax H3 workflows),
Stable Audio 3's VAE encoder fails inside torch SDPA with "cuDNN Frontend error: No valid execution plans built"
(seen 2026-11-26, trisulero-sa3-test). Turning the cuDNN SDPA backend off makes PyTorch fall back to the flash / memory-efficient /
math kernels. Put one node (enable = False) in front of the audio model and one (enable = True) after it, so the rest of the
graph keeps cuDNN attention. Pure pass-through: the audio is not modified.
"""
import torch


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
            print(f"[UE_CudnnSdpToggle] cuDNN SDPA backend {'enabled' if enable else 'disabled'}")
        except Exception as exc:  # old torch without the flag: nothing to toggle
            print(f"[UE_CudnnSdpToggle] could not set cuDNN SDPA ({exc}); continuing")
        return (audio,)


NODE_CLASS_MAPPINGS = {"UE_CudnnSdpToggle": CudnnSdpToggle}
NODE_DISPLAY_NAME_MAPPINGS = {"UE_CudnnSdpToggle": "Audio - cuDNN SDPA toggle (pass-through)"}
