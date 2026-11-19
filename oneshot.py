"""One Shot: turn generated takes into clean hits / SFX (trim the lead-in silence, cap the length, fade the tail, normalise)."""
import numpy as np
import torch


def clean_one_shot(y, sr, max_seconds=1.5, fade_ms=60.0, threshold_db=-42.0, peak_db=-1.0, pre_ms=2.0):
    """y [C, T] float. Start at the first sample above threshold (keeping `pre_ms` before it), keep max_seconds, fade the
    last fade_ms to silence, then scale the peak to peak_db."""
    mono = np.abs(y).max(0)
    thr = 10 ** (threshold_db / 20.0) * max(float(mono.max()), 1e-9)
    idx = np.nonzero(mono > thr)[0]
    start = max(0, int(idx[0]) - int(pre_ms / 1000 * sr)) if len(idx) else 0
    y = y[:, start: start + int(max_seconds * sr)].copy()
    n = y.shape[1]
    f = min(n, int(fade_ms / 1000 * sr))
    if f > 1:
        y[:, n - f:] *= (np.cos(np.linspace(0, np.pi / 2, f, dtype=np.float32)) ** 2)[None]
    a = min(n, int(0.001 * sr))                                  # 1 ms in-fade, no click
    if a > 1:
        y[:, :a] *= np.linspace(0, 1, a, dtype=np.float32)[None]
    peak = float(np.abs(y).max())
    if peak > 1e-9:
        y *= 10 ** (peak_db / 20.0) / peak
    return y


class OneShot:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "max_seconds": ("FLOAT", {"default": 1.5, "min": 0.05, "max": 30.0, "step": 0.05}),
            "fade_ms": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 2000.0, "step": 1.0}),
            "threshold_db": ("FLOAT", {"default": -42.0, "min": -90.0, "max": -6.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, audio, max_seconds, fade_ms, threshold_db):
        w = audio["waveform"].detach().cpu().numpy().astype(np.float32)      # [B, C, T]
        sr = int(audio["sample_rate"])
        outs = [clean_one_shot(w[b], sr, max_seconds, fade_ms, threshold_db) for b in range(w.shape[0])]
        T = max(o.shape[1] for o in outs)
        batch = np.stack([np.pad(o, ((0, 0), (0, T - o.shape[1]))) for o in outs])
        return ({"waveform": torch.from_numpy(np.ascontiguousarray(batch)), "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {"One Shot": OneShot}
NODE_DISPLAY_NAME_MAPPINGS = {"One Shot": "Audio - One Shot (trim, fade, normalise a hit)"}
