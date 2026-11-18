"""DJ-style mixing nodes for ComfyUI AUDIO: beat-aware crossfade (equal-power or bass-swap), overlay-at-time, BPM estimate.

Pure numpy/scipy (no librosa/numba) so it installs anywhere. AUDIO = {"waveform": tensor [B, C, T], "sample_rate": int}.
"""
import math
from fractions import Fraction

import numpy as np
from scipy import signal

try:
    import torch
except ImportError:  # unit tests of the pure functions need no torch
    torch = None


# ---------------------------------------------------------------- pure DSP helpers
def _mono(y):
    return y.mean(axis=0) if y.ndim == 2 else y


def onset_envelope(y, sr, hop=512):
    """Spectral-flux onset strength, one value per `hop` samples."""
    y = _mono(y).astype(np.float32)
    if len(y) < 2048:
        return np.zeros(1, dtype=np.float32)
    _f, _t, Z = signal.stft(y, fs=sr, nperseg=1024, noverlap=1024 - hop, boundary=None, padded=False)
    mag = np.log1p(10.0 * np.abs(Z))
    flux = np.maximum(mag[:, 1:] - mag[:, :-1], 0.0).sum(axis=0)
    flux = np.concatenate([[0.0], flux])
    flux = flux - signal.medfilt(flux, 31) if len(flux) > 31 else flux
    return np.maximum(flux, 0.0).astype(np.float32)


def estimate_tempo(env, sr, hop=512, lo=70.0, hi=180.0, prior=120.0):
    """BPM from the autocorrelation of the onset envelope, with a soft prior against octave errors."""
    if len(env) < 64 or env.max() <= 0:
        return 0.0
    e = env - env.mean()
    ac = signal.correlate(e, e, mode="full")[len(e) - 1:]
    ac = ac / (ac[0] + 1e-9)
    fps = sr / hop
    lags = np.arange(len(ac))
    best, best_score = 0.0, -1e9
    for bpm10 in range(int(lo * 10), int(hi * 10) + 1, 2):
        bpm = bpm10 / 10.0
        lag = fps * 60.0 / bpm
        i = int(round(lag))
        if i < 2 or i + 1 >= len(ac):
            continue
        # a beat period and its multiples should all correlate
        s = ac[i] + 0.5 * (ac[min(2 * i, len(ac) - 1)] if 2 * i < len(ac) else 0.0)
        s *= math.exp(-0.5 * (math.log2(bpm / prior) / 0.9) ** 2)
        if s > best_score:
            best, best_score = bpm, s
    return float(best)


def beat_grid(env, sr, bpm, hop=512):
    """Phase of the beat lattice (seconds of the first beat) and the beat period in seconds for a given BPM."""
    period = 60.0 / bpm
    fps = sr / hop
    n = len(env)
    p = period * fps
    if n < p * 2:
        return 0.0, period
    best_phase, best_score = 0.0, -1.0
    steps = max(8, int(p))
    for k in range(steps):
        phase = k * p / steps
        idx = np.arange(phase, n - 1, p)
        score = float(np.interp(idx, np.arange(n), env).sum())
        if score > best_score:
            best_score, best_phase = score, phase
    return best_phase / fps, period


def detect(y, sr, bpm_hint=0.0, hop=512):
    env = onset_envelope(y, sr, hop)
    bpm = bpm_hint if bpm_hint > 0 else estimate_tempo(env, sr, hop)
    if bpm <= 0:
        return 0.0, 0.0, 0.0
    t0, period = beat_grid(env, sr, bpm, hop)
    return bpm, t0, period


def change_speed(y, rate):
    """Turntable-style speed change (pitch moves with tempo). rate > 1 = faster/shorter."""
    if abs(rate - 1.0) < 1e-4:
        return y
    frac = Fraction(1.0 / rate).limit_denominator(1000)
    out = [signal.resample_poly(ch, frac.numerator, frac.denominator) for ch in y]
    return np.stack(out).astype(np.float32)


def _gains(n, curve):
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if curve == "linear":
        return 1.0 - x, x
    return np.cos(x * math.pi / 2), np.sin(x * math.pi / 2)  # equal power


def _bass_swap(seg_a, seg_b, sr, split_hz=200.0):
    n = seg_a.shape[1]
    sos = signal.butter(4, split_hz, btype="low", fs=sr, output="sos")
    la = signal.sosfiltfilt(sos, seg_a, axis=1)
    lb = signal.sosfiltfilt(sos, seg_b, axis=1)
    ha, hb = seg_a - la, seg_b - lb
    ga, gb = _gains(n, "equal_power")
    # lows: a holds until just past the middle, then b takes over (short smooth swap)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    sw = np.clip((x - 0.45) / 0.10, 0.0, 1.0)
    sw = sw * sw * (3 - 2 * sw)
    return (ha * ga + hb * gb) + (la * (1 - sw) + lb * sw)


def _stereo(y):
    if y.ndim == 1:
        y = y[None]
    if y.shape[0] == 1:
        y = np.repeat(y, 2, axis=0)
    return y[:2].astype(np.float32)


def crossfade(a, b, sr, overlap_s=4.0, curve="bass_swap", beat_align=True, match_tempo=True,
              bpm_a=0.0, bpm_b=0.0, max_stretch=0.08):
    """Join a (earlier) and b (later) with an overlapped transition. Returns (audio [2,T], info dict)."""
    a, b = _stereo(a), _stereo(b)
    info = {}
    ba = detect(a[:, -int(sr * 30):], sr, bpm_a) if beat_align or match_tempo else (0, 0, 0)
    bb = detect(b[:, : int(sr * 30)], sr, bpm_b) if beat_align or match_tempo else (0, 0, 0)
    info["bpm_a"], info["bpm_b"] = round(ba[0], 1), round(bb[0], 1)
    if match_tempo and ba[0] > 0 and bb[0] > 0:
        rate = ba[0] / bb[0]
        for r in (rate, rate * 2, rate / 2):  # allow half/double-time matches
            if abs(r - 1.0) <= max_stretch:
                b = change_speed(b, r)
                bb = detect(b[:, : int(sr * 30)], sr, ba[0])  # re-grid b at a's tempo
                info["stretch"] = round(r, 4)
                break
    la, lb = a.shape[1], b.shape[1]
    ov = int(overlap_s * sr)
    if beat_align and ba[0] > 0 and bb[0] > 0:
        period = ba[2]
        beats = max(1, int(round(overlap_s / period)))
        ov = int(beats * period * sr)
        # beat lattice of a's last 30 s, in absolute samples
        off = max(0, la - int(sr * 30))
        t0a = ba[1]
        want = la - ov
        k = round((want / sr - off / sr - t0a) / period)
        tj = int((off / sr + t0a + k * period) * sr)
        tj = max(0, min(tj, la - 1))
        # trim b's head so its first beat sits on the join
        b = b[:, int(bb[1] * sr):]
        info["beats"] = beats
    else:
        tj = max(0, la - ov)
    ov = min(ov, la - tj, b.shape[1])
    if ov < int(0.05 * sr):
        return np.concatenate([a, b], axis=1), info
    seg_a = a[:, tj: tj + ov]
    seg_b = b[:, :ov]
    if curve == "bass_swap" and ov > 4096:
        mix = _bass_swap(seg_a, seg_b, sr)
    else:
        ga, gb = _gains(ov, "linear" if curve == "linear" else "equal_power")
        mix = seg_a * ga + seg_b * gb
    out = np.concatenate([a[:, :tj], mix, b[:, ov:]], axis=1)
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak
    info["overlap_s"] = round(ov / sr, 3)
    return out.astype(np.float32), info


def overlay_at(base, over, sr, at_s=-1.0, mode="end_at", gain_db=-3.0, fade_in_s=0.0):
    """Layer `over` on `base`. end_at: `over` finishes at at_s (-1 = end of base). start_at: `over` begins at at_s."""
    base, over = _stereo(base), _stereo(over)
    lb, lo = base.shape[1], over.shape[1]
    at = lb if at_s < 0 else int(at_s * sr)
    start = (at - lo) if mode == "end_at" else at
    g = 10 ** (gain_db / 20.0)
    over = over * g
    if fade_in_s > 0:
        n = min(lo, int(fade_in_s * sr))
        over = over.copy()
        over[:, :n] *= np.linspace(0, 1, n, dtype=np.float32)[None]
    total = max(lb, start + lo)
    out = np.zeros((2, total), dtype=np.float32)
    out[:, :lb] += base
    s0 = max(start, 0)
    o0 = s0 - start
    out[:, s0: s0 + (lo - o0)] += over[:, o0:]
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak
    return out


# ---------------------------------------------------------------- ComfyUI nodes
def _to_np(audio):
    w = audio["waveform"][0].detach().cpu().numpy().astype(np.float32)
    return w, int(audio["sample_rate"])


def _to_audio(y, sr):
    return {"waveform": torch.from_numpy(np.ascontiguousarray(y))[None], "sample_rate": sr}


def _match_sr(y, sr, sr_target):
    if sr == sr_target:
        return y
    frac = Fraction(sr_target, sr).limit_denominator(1000)
    return np.stack([signal.resample_poly(ch, frac.numerator, frac.denominator) for ch in y]).astype(np.float32)


class AudioCrossfadeDJ:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio_a": ("AUDIO",), "audio_b": ("AUDIO",),
            "overlap_seconds": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            "curve": (["bass_swap", "equal_power", "linear"], {"default": "bass_swap"}),
            "beat_align": ("BOOLEAN", {"default": True}),
            "match_tempo": ("BOOLEAN", {"default": True}),
            "max_stretch_percent": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 25.0, "step": 0.5}),
        }, "optional": {
            "bpm_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
            "bpm_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, audio_a, audio_b, overlap_seconds, curve, beat_align, match_tempo, max_stretch_percent,
            bpm_a=0.0, bpm_b=0.0):
        a, sra = _to_np(audio_a)
        b, srb = _to_np(audio_b)
        sr = max(sra, srb)
        a, b = _match_sr(_stereo(a), sra, sr), _match_sr(_stereo(b), srb, sr)
        y, info = crossfade(a, b, sr, overlap_seconds, curve, beat_align, match_tempo, bpm_a, bpm_b,
                            max_stretch_percent / 100.0)
        text = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"[AudioCrossfadeDJ] {text}")
        return (_to_audio(y, sr), text)


class AudioOverlayAt:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("AUDIO",), "overlay": ("AUDIO",),
            "mode": (["end_at", "start_at"], {"default": "end_at"}),
            "at_seconds": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 3600.0, "step": 0.01}),
            "gain_db": ("FLOAT", {"default": -3.0, "min": -60.0, "max": 12.0, "step": 0.5}),
            "fade_in_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO",)
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, base, overlay, mode, at_seconds, gain_db, fade_in_seconds):
        b, srb = _to_np(base)
        o, sro = _to_np(overlay)
        sr = max(srb, sro)
        y = overlay_at(_match_sr(_stereo(b), srb, sr), _match_sr(_stereo(o), sro, sr), sr, at_seconds, mode,
                       gain_db, fade_in_seconds)
        return (_to_audio(y, sr),)


class AudioBPM:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("bpm", "info")
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, audio):
        y, sr = _to_np(audio)
        bpm, t0, period = detect(_stereo(y), sr)
        text = f"bpm={bpm:.1f} first_beat={t0:.3f}s period={period:.3f}s"
        print(f"[AudioBPM] {text}")
        return (float(bpm), text)


NODE_CLASS_MAPPINGS = {
    "Audio Crossfade DJ": AudioCrossfadeDJ,
    "Audio Overlay At": AudioOverlayAt,
    "Audio BPM": AudioBPM,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Crossfade DJ": "Audio - Crossfade (DJ, beat-aware)",
    "Audio Overlay At": "Audio - Overlay At Time",
    "Audio BPM": "Audio - BPM Estimate",
}
