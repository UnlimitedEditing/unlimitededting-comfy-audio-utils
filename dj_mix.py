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


def _lattice_best(env, fps, period_frames, steps):
    n = len(env)
    xs = np.arange(n, dtype=np.float64)
    best_phase, best = 0.0, -1.0
    for k in range(steps):
        phase = k * period_frames / steps
        idx = np.arange(phase, n - 1, period_frames)
        sc = float(np.interp(idx, xs, env).sum()) / max(len(idx), 1)
        if sc > best:
            best, best_phase = sc, phase
    return best_phase, best


def beat_grid(env, sr, bpm, hop=512):
    """Phase of the beat lattice (seconds of the first beat) and the beat period in seconds for a given BPM."""
    period = 60.0 / bpm
    fps = sr / hop
    p = period * fps
    if len(env) < p * 2:
        return 0.0, period
    phase, _ = _lattice_best(env, fps, p, max(16, int(p * 2)))
    return phase / fps, period


def refine_tempo(env, sr, coarse, hop=512, span=0.03, step=0.03):
    """Fine BPM search: the lattice that explains the most onset energy over the WHOLE window wins, so a 1% error
    (which would drift half a second over 50 s) is corrected."""
    fps = sr / hop
    best_bpm, best = coarse, -1.0
    for bpm in np.arange(coarse * (1 - span), coarse * (1 + span) + 1e-9, step):
        p = fps * 60.0 / bpm
        if len(env) < p * 2:
            continue
        _, sc = _lattice_best(env, fps, p, max(12, int(p)))
        if sc > best:
            best, best_bpm = sc, float(bpm)
    return best_bpm


def detect(y, sr, bpm_hint=0.0, hop=256):
    env = onset_envelope(y, sr, hop)
    coarse = bpm_hint if bpm_hint > 0 else estimate_tempo(env, sr, hop)
    if coarse <= 0:
        return 0.0, 0.0, 0.0
    bpm = refine_tempo(env, sr, coarse, hop) if bpm_hint <= 0 else coarse
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



# ---------------------------------------------------------------- DJ transition styles
STYLE_NAMES = ["equal_power", "bass_swap", "eq_3band", "filter_sweep", "echo_out", "reverb_out", "loop_roll", "linear"]


def _smooth(x, lo, hi):
    """0 below lo, 1 above hi, smoothstep between."""
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3 - 2 * t)).astype(np.float32)


def _split3(y, sr, f1=200.0, f2=2500.0):
    lo = signal.sosfiltfilt(signal.butter(4, f1, btype="low", fs=sr, output="sos"), y, axis=1)
    hi = signal.sosfiltfilt(signal.butter(4, f2, btype="high", fs=sr, output="sos"), y, axis=1)
    return lo, y - lo - hi, hi


def _eq_3band(seg_a, seg_b, sr):
    """A DJ's three-knob blend: b's highs come in first, mids ride across the middle, the bass swaps last (and a's
    highs go early, mids mid, bass last), so the two kicks never fight for the low end."""
    n = seg_a.shape[1]
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    la, ma, ha = _split3(seg_a, sr)
    lb, mb, hb = _split3(seg_b, sr)
    low_b = _smooth(x, 0.46, 0.54)
    mid_b = _smooth(x, 0.25, 0.75)
    hi_b = _smooth(x, 0.0, 0.5)
    return (la * (1 - low_b) + lb * low_b) + (ma * (1 - mid_b) + mb * mid_b) + (ha * (1 - hi_b) + hb * hi_b)


def _sweep(y, sr, kind, f0, f1, steps=14):
    """Time-varying high/low-pass: cutoff glides exponentially f0 -> f1 across y. Blends a bank of fixed filters."""
    n = y.shape[1]
    cuts = np.geomspace(min(f0, f1), max(f0, f1), steps)
    bank = [signal.sosfilt(signal.butter(2, min(c, sr * 0.45), btype=kind, fs=sr, output="sos"), y, axis=1) for c in cuts]
    pos = np.linspace(np.log(f0), np.log(f1), n)
    idx = np.interp(pos, np.log(cuts), np.arange(steps))
    lo = np.clip(np.floor(idx).astype(int), 0, steps - 2)
    w = (idx - lo).astype(np.float32)
    out = np.zeros_like(y)
    stack = np.stack(bank)  # [steps, C, T]
    t = np.arange(n)
    out = stack[lo, :, t].T * (1 - w)[None] + stack[lo + 1, :, t].T * w[None]
    return out.astype(np.float32)


def _filter_sweep(seg_a, seg_b, sr):
    """Outgoing track is high-passed up and out (thinning to a hiss), incoming opens from a muffled low-pass."""
    n = seg_a.shape[1]
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    a = _sweep(seg_a, sr, "highpass", 40.0, 3500.0) * (1 - _smooth(x, 0.55, 1.0))[None]
    b = _sweep(seg_b, sr, "lowpass", 250.0, 18000.0) * _smooth(x, 0.0, 0.45)[None]
    return a + b


def _delay_tail(y, sr, delay_s, taps=5, feedback=0.55):
    d = max(1, int(delay_s * sr))
    out = np.zeros((y.shape[0], y.shape[1] + d * taps), dtype=np.float32)
    for k in range(1, taps + 1):
        out[:, d * k: d * k + y.shape[1]] += y * (feedback ** k)
    return out


def _reverb_tail(y, sr, seconds=2.5):
    n = int(seconds * sr)
    rng = np.random.default_rng(7)
    ir = rng.standard_normal((2, n)).astype(np.float32) * np.exp(-np.linspace(0, 7.0, n))[None].astype(np.float32)
    ir = signal.sosfilt(signal.butter(2, 5000.0, btype="low", fs=sr, output="sos"), ir, axis=1)
    wet = np.stack([signal.fftconvolve(y[c], ir[c]) for c in range(2)]).astype(np.float32)
    return wet * 0.05


def _loop_roll(a, tj_beats_start, period_s, sr):
    """Stutter the last bar before the cut: loops of 1, 1/2, 1/4, 1/8 beat, each filling one beat."""
    p = int(period_s * sr)
    base = a[:, tj_beats_start: tj_beats_start + 4 * p]
    if base.shape[1] < 4 * p:
        return a[:, tj_beats_start:]
    parts = [base[:, :p]]
    for k, div in enumerate((2, 4, 8), start=1):
        seg = base[:, k * p: k * p + p // div]
        reps = int(np.ceil(p / max(seg.shape[1], 1)))
        parts.append(np.tile(seg, (1, reps))[:, :p])
    return np.concatenate(parts, axis=1)



def crossfade(a, b, sr, overlap_s=4.0, curve="bass_swap", beat_align=True, match_tempo=True,
              bpm_a=0.0, bpm_b=0.0, max_stretch=0.08, trans=None, trans_gain_db=-6.0, trans_ends="overlap_end"):
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
    period_s = ba[2] if (beat_align and ba[0] > 0) else max(ov / sr / 4.0, 0.25)
    tail = None
    if curve == "loop_roll" and beat_align and ba[0] > 0 and ov >= int(4 * period_s * sr * 0.9):
        # stutter the last bar of a, then cut straight to b on its first beat
        p = int(period_s * sr)
        start = max(0, tj + ov - 4 * p)
        rolled = _loop_roll(a, start, period_s, sr)
        out = np.concatenate([a[:, :start], rolled, b[:, :]], axis=1)
        info["overlap_s"] = 0.0
        info["style"] = "loop_roll"
        peak = float(np.abs(out).max())
        return (out / peak if peak > 1.0 else out).astype(np.float32), info
    if ov > 4096 and curve == "bass_swap":
        mix = _bass_swap(seg_a, seg_b, sr)
    elif ov > 8192 and curve == "eq_3band":
        mix = _eq_3band(seg_a, seg_b, sr)
    elif ov > 8192 and curve == "filter_sweep":
        mix = _filter_sweep(seg_a, seg_b, sr)
    elif curve in ("echo_out", "reverb_out"):
        # a is cut quickly (equal-power fast fade) while its echo/reverb tail keeps ringing under b
        x = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        fast_out = np.cos(np.clip(x / 0.5, 0, 1) * math.pi / 2)[None]
        dry_a = seg_a * fast_out
        gb = np.sin(np.clip((x - 0.1) / 0.8, 0, 1) * math.pi / 2)[None]
        mix = dry_a + seg_b * gb
        wet = _delay_tail(dry_a, sr, period_s * 0.75) if curve == "echo_out" else _reverb_tail(dry_a, sr)
        tail = wet
    else:
        ga, gb = _gains(ov, "linear" if curve == "linear" else "equal_power")
        mix = seg_a * ga + seg_b * gb
    info["style"] = curve if curve in STYLE_NAMES else "equal_power"
    out = np.concatenate([a[:, :tj], mix, b[:, ov:]], axis=1)
    if tail is not None:
        need = tj + tail.shape[1]
        if need > out.shape[1]:
            out = np.concatenate([out, np.zeros((2, need - out.shape[1]), dtype=np.float32)], axis=1)
        out[:, tj: tj + tail.shape[1]] += tail
    if trans is not None:
        end_at = tj + (ov // 2 if trans_ends == "overlap_midpoint" else ov)
        fade = 0.8 * trans.shape[-1] / sr
        out = overlay_at(out, trans, sr, end_at / sr, "end_at", trans_gain_db, fade)
        info["transition"] = trans_ends
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak
    info["overlap_s"] = round(ov / sr, 3)
    return out.astype(np.float32), info



def bar_phase(env, sr, bpm, hop=256, beats_per_bar=4):
    """Seconds (within the first bar) of the strongest bar line for a given tempo, plus the bar period in seconds."""
    fps = sr / hop
    p = fps * 60.0 / bpm * beats_per_bar
    if len(env) < p * 2:
        return 0.0, 60.0 / bpm * beats_per_bar
    phase, _ = _lattice_best(env, fps, p, max(16, int(p)))
    return phase / fps, 60.0 / bpm * beats_per_bar


def fit_tempo_to(env, sr, target_bpm, hop=256, span=0.10, step=0.002):
    """Which native tempo near target_bpm (or at half/double) makes the layer's hits sit on a beat lattice best?
    Returns (native_bpm, score). Scans a fine grid, so triplet/dotted detector confusions do not matter."""
    fps = sr / hop
    best, best_bpm = -1.0, target_bpm
    for mult, w in ((1.0, 1.0),):
        c0 = target_bpm * mult
        for bpm in np.arange(c0 * (1 - span), c0 * (1 + span) + 1e-9, c0 * step):
            p = fps * 60.0 / bpm
            if len(env) < p * 2:
                continue
            _, sc = _lattice_best(env, fps, p, max(12, int(p)))
            if sc * w > best:
                best, best_bpm = sc * w, float(bpm)
    return best_bpm, best


def place_layer(base, layer, sr, start_s, end_s, gain_db=-3.0, fade_in_s=2.0, fade_out_s=2.0, align=True,
                match_tempo=True, max_stretch=0.10, loop=True, bpm_base=0.0):
    """Lay a stem over `base` between start_s and end_s, locked to base's beat grid and bar lines.
    The layer's tempo is FITTED to the base tempo (scan within +-10%, half/double allowed) and it is sped up/down
    by the needed factor (if within max_stretch), its bar line is put on a base bar line, it is looped to fill the
    region if it is shorter, and faded in/out. Returns (audio [2,T], info dict)."""
    base, layer = _stereo(base), _stereo(layer)
    info = {}
    hop = 256
    env_b = onset_envelope(base[:, : int(sr * 40)], sr, hop)
    bpm_b = bpm_base if bpm_base > 0 else refine_tempo(env_b, sr, estimate_tempo(env_b, sr, hop), hop)
    info["bpm_base"] = round(bpm_b, 2)
    ts, te = float(start_s), float(end_s)
    if align and bpm_b > 0:
        bar0_b, bar_b = bar_phase(env_b, sr, bpm_b, hop)
        env_l = onset_envelope(layer[:, : int(sr * 30)], sr, hop)
        c, _ = fit_tempo_to(env_l, sr, bpm_b, hop)
        info["bpm_layer_fit"] = round(c, 2)
        rate = bpm_b / c
        if match_tempo and abs(rate - 1.0) <= max_stretch and abs(rate - 1.0) > 0.0005:
            layer = change_speed(layer, rate)
            env_l = onset_envelope(layer[:, : int(sr * 30)], sr, hop)
            info["stretch"] = round(rate, 4)
        bar0_l, _ = bar_phase(env_l, sr, bpm_b if match_tempo else c, hop)
        layer = layer[:, int(bar0_l * sr):]                 # the layer now starts on its own bar line
        snap = lambda t: bar0_b + round((t - bar0_b) / bar_b) * bar_b
        ts, te = max(0.0, snap(ts)), snap(te)
        info["start_s"], info["end_s"] = round(ts, 3), round(te, 3)
    n0, n1 = int(ts * sr), int(te * sr)
    want = max(0, n1 - n0)
    if want == 0:
        return base, info
    if layer.shape[1] < want and loop:
        reps = int(np.ceil(want / layer.shape[1]))
        layer = np.tile(layer, (1, reps))
        info["looped"] = reps
    layer = layer[:, :want].copy()
    g = 10 ** (gain_db / 20.0)
    fi, fo = min(int(fade_in_s * sr), want // 2), min(int(fade_out_s * sr), want // 2)
    if fi > 0:
        layer[:, :fi] *= np.sin(np.linspace(0, math.pi / 2, fi, dtype=np.float32))[None]
    if fo > 0:
        layer[:, -fo:] *= np.cos(np.linspace(0, math.pi / 2, fo, dtype=np.float32))[None]
    total = max(base.shape[1], n0 + want)
    out = np.zeros((2, total), dtype=np.float32)
    out[:, : base.shape[1]] += base
    out[:, n0: n0 + want] += layer * g
    peak = float(np.abs(out).max())
    if peak > 1.0:
        out = out / peak
    return out, info


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
            "curve": (STYLE_NAMES, {"default": "equal_power"}),
            "beat_align": ("BOOLEAN", {"default": True}),
            "match_tempo": ("BOOLEAN", {"default": True}),
            "max_stretch_percent": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 25.0, "step": 0.5}),
        }, "optional": {
            "transition": ("AUDIO",),
            "transition_gain_db": ("FLOAT", {"default": -6.0, "min": -60.0, "max": 12.0, "step": 0.5}),
            "transition_ends_at": (["overlap_end", "overlap_midpoint"], {"default": "overlap_end"}),
            "bpm_a": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
            "bpm_b": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 300.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, audio_a, audio_b, overlap_seconds, curve, beat_align, match_tempo, max_stretch_percent,
            transition=None, transition_gain_db=-6.0, transition_ends_at="overlap_end", bpm_a=0.0, bpm_b=0.0):
        a, sra = _to_np(audio_a)
        b, srb = _to_np(audio_b)
        sr = max(sra, srb)
        a, b = _match_sr(_stereo(a), sra, sr), _match_sr(_stereo(b), srb, sr)
        tr = None
        if transition is not None:
            t, srt = _to_np(transition)
            tr = _match_sr(_stereo(t), srt, sr)
        y, info = crossfade(a, b, sr, overlap_seconds, curve, beat_align, match_tempo, bpm_a, bpm_b,
                            max_stretch_percent / 100.0, tr, transition_gain_db, transition_ends_at)
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



class AudioLayerPlace:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base": ("AUDIO",), "layer": ("AUDIO",),
            "start_seconds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            "end_seconds": ("FLOAT", {"default": 24.0, "min": 0.0, "max": 3600.0, "step": 0.01}),
            "gain_db": ("FLOAT", {"default": -3.0, "min": -60.0, "max": 12.0, "step": 0.5}),
            "fade_in_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            "fade_out_seconds": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 60.0, "step": 0.1}),
            "beat_lock": ("BOOLEAN", {"default": True}),
            "max_stretch_percent": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 25.0, "step": 0.5}),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, base, layer, start_seconds, end_seconds, gain_db, fade_in_seconds, fade_out_seconds, beat_lock,
            max_stretch_percent):
        b, srb = _to_np(base)
        l, srl = _to_np(layer)
        sr = max(srb, srl)
        y, info = place_layer(_match_sr(_stereo(b), srb, sr), _match_sr(_stereo(l), srl, sr), sr, start_seconds,
                              end_seconds, gain_db, fade_in_seconds, fade_out_seconds, beat_lock, True,
                              max_stretch_percent / 100.0)
        text = ", ".join(f"{k}={v}" for k, v in info.items())
        print(f"[AudioLayerPlace] {text}")
        return (_to_audio(y, sr), text)


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
    "Audio Layer Place": AudioLayerPlace,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Audio Crossfade DJ": "Audio - Crossfade (DJ, beat-aware)",
    "Audio Overlay At": "Audio - Overlay At Time",
    "Audio BPM": "Audio - BPM Estimate",
    "Audio Layer Place": "Audio - Place Layer (beat-locked)",
}
