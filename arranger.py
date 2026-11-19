"""Drum arranger: turn a small bank of drum loops at different intensities into a drum track that follows a song structure.

The bank (each loop whole bars at the same BPM): `low` (sparse), `mid` (steady groove), `high` (full energy), `fill` (a build /
fill that ends on a crash) and `breakdown` (no kick). An energy curve is built from a structure string, e.g.
    intro:8:0.2-0.4,build:8:0.4-0.9,drop:16:1.0,break:8:0.15,build:8:0.3-0.9,drop:16:1.0,outro:8:0.6-0.1
(name:bars:energy or name:bars:from-to). For every bar the sequencer picks the clip that fits that bar's energy, keeps phrases
continuous (it steps through a clip bar by bar), and dresses the boundaries: a fill into a drop, a crescendo and filter
opening on builds, and one last "tick" that reverbs out into the next section. Pure numpy/scipy.
"""
import math
import re

import numpy as np
from scipy import signal

try:
    import torch
except ImportError:
    torch = None

from . import dj_mix as _dj  # noqa: E402  (reuse _reverb_tail, _stereo, _to_np, _to_audio)

TIERS = ("low", "mid", "high")
TIER_CENTER = {"low": 0.2, "mid": 0.55, "high": 0.95}


# ---------------------------------------------------------------- structure
def parse_structure(text):
    secs = []
    for tok in re.split(r"[,;\n]+", text.strip()):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) < 3:
            raise ValueError(f"bad section '{tok}': expected name:bars:energy (e.g. build:8:0.4-0.9)")
        name, bars, en = parts[0].strip().lower(), int(float(parts[1])), parts[2].strip()
        m = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", en)
        e0, e1 = (float(m.group(1)), float(m.group(2))) if m else (float(en), float(en))
        secs.append({"name": name, "bars": max(1, bars), "e0": e0, "e1": e1})
    if not secs:
        raise ValueError("empty structure")
    return secs


def bar_energies(secs):
    out = []
    for si, s in enumerate(secs):
        for i in range(s["bars"]):
            f = i / max(s["bars"] - 1, 1)
            out.append((si, i, s["e0"] + (s["e1"] - s["e0"]) * f))
    return out


# ---------------------------------------------------------------- bar bank
def _slice_bars(y, bar_n):
    n = max(1, int(round(y.shape[1] / bar_n)))
    bars = []
    for i in range(n):
        seg = y[:, i * bar_n: (i + 1) * bar_n]
        if seg.shape[1] < bar_n:
            seg = np.pad(seg, ((0, 0), (0, bar_n - seg.shape[1])))
        bars.append(seg.astype(np.float32))
    return bars


def _xfade(a, b, n):
    n = min(n, a.shape[1], b.shape[1])
    if n < 2:
        return np.concatenate([a, b], axis=1)
    t = np.linspace(0, 1, n, dtype=np.float32)[None]
    mid = a[:, -n:] * np.cos(t * math.pi / 2) + b[:, :n] * np.sin(t * math.pi / 2)
    return np.concatenate([a[:, :-n], mid, b[:, n:]], axis=1)


# ---------------------------------------------------------------- dressings
def _hp_sweep(y, sr, f_from, f_to, steps=12):
    """High-pass whose cutoff glides f_from -> f_to over y (a filter opening is f_from high -> f_to low)."""
    n = y.shape[1]
    cuts = np.geomspace(min(f_from, f_to), max(f_from, f_to), steps)
    bank = [signal.sosfilt(signal.butter(2, c, btype="highpass", fs=sr, output="sos"), y, axis=1) for c in cuts]
    pos = np.linspace(np.log(f_from), np.log(f_to), n)
    idx = np.interp(pos, np.log(cuts), np.arange(steps))
    lo = np.clip(np.floor(idx).astype(int), 0, steps - 2)
    w = (idx - lo).astype(np.float32)
    stack = np.stack(bank)
    t = np.arange(n)
    return (stack[lo, :, t].T * (1 - w)[None] + stack[lo + 1, :, t].T * w[None]).astype(np.float32)


def _crescendo(seg, sr, g0=0.8, g1=1.2, filt=True):
    n = seg.shape[1]
    ramp = np.linspace(g0, g1, n, dtype=np.float32)[None]
    out = seg * ramp
    if filt:
        out = 0.5 * out + 0.5 * _hp_sweep(out, sr, 250.0, 30.0)   # keep half the dry kick so the build never drops out
    return out


def _tick_reverb(track, sr, bar_start, bar_n, beat_n, tail_s=2.4):
    """Last bar of a section: keep three beats, thin the last beat down to the final 16th 'tick' (its high band), and let
    that tick ring out in reverb over the start of the next section. The tail may run past the end of `track`."""
    end = bar_start + bar_n
    last_beat = end - beat_n
    tick_n = max(1, beat_n // 4)
    tick = track[:, end - tick_n: end].copy()
    tick = signal.sosfiltfilt(signal.butter(2, 2500.0, btype="highpass", fs=sr, output="sos"), tick, axis=1) * 1.6
    track[:, last_beat:end] *= 0.10                      # the last beat drops away, leaving space for the tick
    wet = _dj._reverb_tail(tick, sr, tail_s) * 9.0
    need = end - tick_n + wet.shape[1]
    if need > track.shape[1]:
        track = np.concatenate([track, np.zeros((2, need - track.shape[1]), dtype=np.float32)], axis=1)
    track[:, end - tick_n: end] += tick
    track[:, end - tick_n: end - tick_n + wet.shape[1]] += wet
    return track


def _reverse_swell(track, sr, bar_start, bar_n, frac=0.5):
    n = int(bar_n * frac)
    end = bar_start + bar_n
    seg = track[:, end - n: end].copy()
    rev = seg[:, ::-1].copy()
    rev *= np.linspace(0.0, 1.0, n, dtype=np.float32)[None] ** 2
    wet = _dj._reverb_tail(seg, sr, 1.2)[:, :n] * 4.0
    wet = wet[:, ::-1] * np.linspace(0.0, 1.0, n, dtype=np.float32)[None] ** 2
    track[:, end - n: end] = rev * 0.8 + wet * 0.6
    return track


# ---------------------------------------------------------------- the arranger
def arrange(tiers, sr, bpm, structure, beats_per_bar=4, variation=0.35, seed=0, transitions="auto"):
    """tiers: dict of name -> stereo float array; needs at least one of low/mid/high. Returns (audio [2,T], plan list)."""
    rng = np.random.default_rng(seed)
    bar_s = beats_per_bar * 60.0 / bpm
    bar_n = int(round(bar_s * sr))
    beat_n = int(round(60.0 / bpm * sr))
    ref = tiers.get("mid") if tiers.get("mid") is not None else next(v for v in tiers.values() if v is not None)
    ref_rms = float(np.sqrt(np.mean(_dj._stereo(ref) ** 2))) + 1e-9

    def _norm(v):
        v = _dj._stereo(v)
        r = float(np.sqrt(np.mean(v ** 2))) + 1e-9
        return v * min(ref_rms / r, 6.0)            # never boost a near-silent clip by more than ~15 dB

    bank = {k: _slice_bars(_norm(v), bar_n) for k, v in tiers.items() if v is not None}
    avail = [t for t in TIERS if t in bank]
    if not avail:
        raise ValueError("need at least one of low / mid / high")

    def nearest(e):
        return min(avail, key=lambda t: abs(TIER_CENTER[t] - e))

    secs = parse_structure(structure)
    be = bar_energies(secs)
    ptr = {k: 0 for k in bank}
    last_pick = None
    bars_out, plan = [], []
    for gi, (si, bi, e) in enumerate(be):
        sec = secs[si]
        if sec["name"] in ("break", "breakdown", "bridge") and "breakdown" in bank:
            tier = "breakdown"
        else:
            e_j = float(np.clip(e + rng.normal(0, 0.12 * variation), 0, 1))
            tier = nearest(e_j)
        is_last = bi == sec["bars"] - 1
        nxt = secs[si + 1] if si + 1 < len(secs) else None
        want_fill = (is_last and "fill" in bank and nxt is not None and (nxt["name"] in ("drop", "chorus", "main") or nxt["e0"] > e + 0.25))
        if want_fill:
            src_bars, k = bank["fill"], "fill"
            i = len(src_bars) - 1
        else:
            src_bars = bank[tier]
            k = tier
            if bi == 0 or not last_pick or last_pick[0] != k:
                i = 0                                            # start the clip at its first bar: phrases begin on the one
            else:
                i = (last_pick[1] + 1) % len(src_bars)           # play it through
                if i == 0 and rng.random() < variation * 0.5:    # at a turnaround, occasionally take a different bar
                    i = int(rng.integers(1, len(src_bars))) if len(src_bars) > 1 else 0
        last_pick = (k, i)
        bar = src_bars[i].copy()
        gain = 0.50 + 0.50 * float(e)
        bars_out.append(bar * gain)
        plan.append({"bar": gi, "section": sec["name"], "energy": round(float(e), 2), "clip": f"{k}[{i}]"})
    # assemble on the exact bar grid; a 1 ms fade either side of a jump (where the source clip changes) avoids a click
    track = np.zeros((2, len(bars_out) * bar_n), dtype=np.float32)
    for i, bar in enumerate(bars_out):
        track[:, i * bar_n: (i + 1) * bar_n] = bar
    fade_n = max(2, int(0.001 * sr))
    for i in range(1, len(bars_out)):
        k0, i0 = plan[i - 1]["clip"].split("[")
        k1, i1 = plan[i]["clip"].split("[")
        if not (k0 == k1 and int(i1[:-1]) == (int(i0[:-1]) + 1) % len(bank[k1])):
            s0 = i * bar_n
            track[:, s0 - fade_n: s0] *= np.linspace(1, 0, fade_n, dtype=np.float32)[None]
            track[:, s0: s0 + fade_n] *= np.linspace(0, 1, fade_n, dtype=np.float32)[None]
    # dressings per section boundary
    tr = transitions
    starts = np.cumsum([0] + [s["bars"] for s in secs])
    for si, sec in enumerate(secs):
        b0, b1 = starts[si], starts[si + 1]
        nxt = secs[si + 1] if si + 1 < len(secs) else None
        is_build = sec["name"] in ("build", "buildup", "riser") or (sec["e1"] > sec["e0"] + 0.25)
        if tr in ("auto", "crescendo", "mix") and is_build:
            n_b = min(2, sec["bars"]) * bar_n
            a = b1 * bar_n - n_b
            track[:, a: b1 * bar_n] = _crescendo(track[:, a: b1 * bar_n], sr)
            plan[b1 - 1]["dress"] = "crescendo + filter opening"
        if nxt is not None:
            before_break = nxt["name"] in ("break", "breakdown", "bridge")
            before_drop = nxt["name"] in ("drop", "chorus", "main") or nxt["e0"] > sec["e1"] + 0.25
            if tr in ("auto", "tick_reverb", "mix") and (before_break or before_drop or tr == "tick_reverb"):
                track = _tick_reverb(track, sr, (b1 - 1) * bar_n, bar_n, beat_n)
                plan[b1 - 1]["dress"] = (plan[b1 - 1].get("dress", "") + " + tick reverb-out").strip(" +")
            elif tr == "reverse_swell":
                track = _reverse_swell(track, sr, (b1 - 1) * bar_n, bar_n)
                plan[b1 - 1]["dress"] = "reverse swell"
    peak = float(np.abs(track).max())
    if peak > 0.98:
        track = track / peak * 0.98
    return track.astype(np.float32), plan


# ---------------------------------------------------------------- ComfyUI node
class DrumArranger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "mid": ("AUDIO",),
            "bpm": ("FLOAT", {"default": 124.0, "min": 40.0, "max": 240.0, "step": 0.01}),
            "structure": ("STRING", {"multiline": True, "default":
                          "intro:8:0.2-0.4,build:8:0.4-0.9,drop:16:1.0,break:8:0.15,build:8:0.3-0.9,drop:16:1.0,outro:8:0.6-0.1"}),
            "transitions": (["auto", "tick_reverb", "crescendo", "reverse_swell", "none"], {"default": "auto"}),
            "variation": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 31 - 1}),
            "beats_per_bar": ("INT", {"default": 4, "min": 2, "max": 8}),
        }, "optional": {
            "low": ("AUDIO",), "high": ("AUDIO",), "fill": ("AUDIO",), "breakdown": ("AUDIO",),
        }}

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "plan")
    FUNCTION = "run"
    CATEGORY = "audio/dj"

    def run(self, mid, bpm, structure, transitions, variation, seed, beats_per_bar, low=None, high=None, fill=None, breakdown=None):
        tiers, sr = {}, None
        for name, a in (("low", low), ("mid", mid), ("high", high), ("fill", fill), ("breakdown", breakdown)):
            if a is None:
                continue
            y, s = _dj._to_np(a)
            sr = sr or s
            tiers[name] = _dj._match_sr(_dj._stereo(y), s, sr)
        track, plan = arrange(tiers, sr, bpm, structure, beats_per_bar, variation, seed, transitions)
        text = "; ".join(f"{p['bar']}:{p['section']}@{p['energy']}={p['clip']}{(' ['+p['dress']+']') if p.get('dress') else ''}" for p in plan)
        print(f"[DrumArranger] {len(plan)} bars, {track.shape[1] / sr:.1f}s")
        return (_dj._to_audio(track, sr), text)


NODE_CLASS_MAPPINGS = {"Drum Arranger": DrumArranger}
NODE_DISPLAY_NAME_MAPPINGS = {"Drum Arranger": "Audio - Drum Arranger (build / release by structure)"}
