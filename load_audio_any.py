import os
import re
import subprocess
import tempfile
import urllib.request

try:
    import folder_paths
except ImportError:
    folder_paths = None

# Ported directly from this org's ComfyUI-HiggsV3Glue (HiggsV3VoicePreset's
# _resolve_reference_audio/_unmangle_telegram_file_url/_decode_audio_via_ffmpeg),
# a proven-working pattern on real Graydient jobs, rather than reinventing a
# weaker version. Two things confirmed live there that a naive http-prefix
# check misses:
#   1. Graydient sometimes hands back a Telegram voice-reply reference with
#      every ':'/'/' stripped (e.g. "init_audio__httpsapi.telegram.org...oga")
#      instead of a real URL or a local filename -- reconstructable via a
#      narrow, known-shape regex, not fetchable as-is.
#   2. torchaudio.load() routes through torchcodec's AudioDecoder on recent
#      torchaudio versions, which is broken on at least some Graydient
#      instances ("_AudioDecoder() takes no arguments") -- decode via an
#      ffmpeg subprocess instead, sidestepping torchcodec entirely.

_MANGLED_TELEGRAM_FILE_RE = re.compile(
    r'^(?:[a-z_]+__)?(https?)api\.telegram\.orgfilebot(\d+)([A-Za-z0-9_-]+?)'
    r'(voice|photo|video_note|video|audio|music|document|animation|sticker)file(\d+)\.([a-z0-9]+)$',
    re.IGNORECASE,
)


def _unmangle_telegram_file_url(value):
    m = _MANGLED_TELEGRAM_FILE_RE.match(value)
    if not m:
        return None
    scheme, bot_id, bot_secret, media_kind, file_id, ext = m.groups()
    return f"{scheme}://api.telegram.org/file/bot{bot_id}:{bot_secret}/{media_kind}/file_{file_id}.{ext}"


def _download(url, dest_path):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        f.write(resp.read())


def _decode_audio_via_ffmpeg(path, sample_rate=32000, channels=1):
    """Decode to a [channels, samples] waveform tensor via ffmpeg subprocess (not
    torchaudio.load, see module docstring). Raw PCM output, not WAV, since ffmpeg's
    WAV header on a pipe has a placeholder frame count.

    The defaults (mono, 32 kHz, 16-bit) are what every voice-reference caller was
    built and confirmed against -- leave them alone. Music callers (stem
    separation, mastering) pass channels=2 and a full-band rate, and get float32
    PCM so nothing is requantized on the way in."""
    import numpy as np
    import torch

    fmt, codec, dtype, scale = (("s16le", "pcm_s16le", "<i2", 32768.0) if channels == 1
                                else ("f32le", "pcm_f32le", "<f4", 1.0))
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", fmt, "-acodec", codec,
        "-ar", str(sample_rate), "-ac", str(channels), "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")

    arr = np.frombuffer(proc.stdout, dtype=dtype).astype("float32") / scale
    # ffmpeg interleaves channels: L R L R ... -> [channels, samples]
    waveform = torch.from_numpy(arr.reshape(-1, channels).T.copy())
    return waveform, sample_rate


def _resolve_reference_audio(value, source_label, sample_rate=32000, channels=1):
    value = (value or "").strip()
    if not value:
        return None

    if value.startswith("http://") or value.startswith("https://"):
        print(f"[LoadAudioAny] {source_label}={value!r} -> treating as URL")
        suffix = os.path.splitext(value.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(value, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path, sample_rate, channels)

    reconstructed = _unmangle_telegram_file_url(value)
    if reconstructed is not None:
        print(f"[LoadAudioAny] {source_label}={value!r} -> reconstructed mangled Telegram URL {reconstructed!r}")
        suffix = os.path.splitext(reconstructed.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(reconstructed, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path, sample_rate, channels)

    candidates = [value]
    if folder_paths is not None:
        try:
            candidates.append(os.path.join(folder_paths.get_input_directory(), value))
        except Exception:
            pass

    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"[LoadAudioAny] {source_label}={value!r} -> found local file {candidate!r}")
            return _decode_audio_via_ffmpeg(candidate, sample_rate, channels)

    raise ValueError(
        f"{source_label}={value!r} is not a valid http(s) URL, a reconstructable "
        "Telegram file reference, or an existing local file (checked ComfyUI's "
        f"input/ directory and the literal value). Tried: {candidates!r}"
    )


class LoadAudioAny:
    """
    Loads AUDIO from up to three STRING inputs (audio_source, audio_source_alt,
    audio_source_filename) -- checked in order, first non-empty wins. Each may be
    a real http(s) URL, a mangled Telegram file reference, or a bare local
    filename already staged in ComfyUI's input/ directory.

    Three separate inputs exist, not one, for the same reason
    HiggsV3VoicePreset's custom_audio_url/custom_audio_url_alt/
    custom_audio_filename split does: it's unclear which local_field name a
    given Graydient submission path actually populates (init_audio vs
    init_audio_url vs init_audio_filename), so map all three defensively
    rather than guessing and silently dropping whichever one gets used.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_source": ("STRING", {"multiline": False, "default": ""}),
                "audio_source_alt": ("STRING", {"multiline": False, "default": ""}),
                "audio_source_filename": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    def run(self, audio_source="", audio_source_alt="", audio_source_filename=""):
        for value, label in (
            (audio_source, "audio_source"),
            (audio_source_alt, "audio_source_alt"),
            (audio_source_filename, "audio_source_filename"),
        ):
            result = _resolve_reference_audio(value, label)
            if result is not None:
                waveform, sample_rate = result
                return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

        raise RuntimeError(
            "LoadAudioAny: all three inputs (audio_source/audio_source_alt/"
            "audio_source_filename) were empty."
        )


class LoadAudioAnyOptional(LoadAudioAny):
    """
    Same resolution logic as LoadAudioAny (URL / mangled-Telegram / local-file, in
    that order across the same three inputs), for graphs where a reference/source
    clip is genuinely OPTIONAL (e.g. AuK's instruct-TTS mode, which has no
    reference audio at all) -- unlike LoadAudioAny, which always raises when all
    three inputs are empty (correct for every OTHER caller in this org's
    workflows, where audio is required; added here as a separate class rather than
    changing LoadAudioAny's behavior, so no existing deployed workflow is affected).

    Returns (None,) instead of raising when all three inputs are empty. This is
    safe to wire into any optional AUDIO input -- see e.g. AuK's own
    AuKGenerateEdit.input_audio, whose node code explicitly handles
    `if audio is None: return None` for exactly this case.
    """

    def run(self, audio_source="", audio_source_alt="", audio_source_filename=""):
        for value, label in (
            (audio_source, "audio_source"),
            (audio_source_alt, "audio_source_alt"),
            (audio_source_filename, "audio_source_filename"),
        ):
            result = _resolve_reference_audio(value, label)
            if result is not None:
                waveform, sample_rate = result
                return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)

        return (None,)


class LoadAudioAnyStereo:
    """
    Same resolution logic as LoadAudioAny (URL / mangled-Telegram / local-file,
    first non-empty of the same three inputs), but decodes STEREO float32 at a
    chosen sample rate instead of LoadAudioAny's mono 32 kHz.

    LoadAudioAny's mono/32 kHz decode is right for its voice-reference callers
    and badly wrong for music: stem separators (Mel-Band RoFormer, Demucs) rely
    on the stereo image and on content above 16 kHz, and got neither -- audible
    vocal artifacts and a disturbed instrumental on the first stems-vocals job,
    and a hard crash in set-soft's Demucs node, whose mono branch reads a
    `demixer.ch` attribute DemixerDemucs never sets. Added as a separate class,
    like LoadAudioAnyOptional, so no deployed workflow changes behavior.

    Mono sources come out as two identical channels (ffmpeg upmixes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_source": ("STRING", {"multiline": False, "default": ""}),
                "audio_source_alt": ("STRING", {"multiline": False, "default": ""}),
                "audio_source_filename": ("STRING", {"multiline": False, "default": ""}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 50}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    def run(self, audio_source="", audio_source_alt="", audio_source_filename="", sample_rate=44100):
        for value, label in (
            (audio_source, "audio_source"),
            (audio_source_alt, "audio_source_alt"),
            (audio_source_filename, "audio_source_filename"),
        ):
            result = _resolve_reference_audio(value, label, sample_rate, channels=2)
            if result is not None:
                waveform, sr = result
                return ({"waveform": waveform.unsqueeze(0), "sample_rate": sr},)

        raise RuntimeError(
            "LoadAudioAnyStereo: all three inputs (audio_source/audio_source_alt/"
            "audio_source_filename) were empty."
        )


NODE_CLASS_MAPPINGS = {
    "Load Audio Any": LoadAudioAny,
    "Load Audio Any Optional": LoadAudioAnyOptional,
    "Load Audio Any Stereo": LoadAudioAnyStereo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Audio Any": "Audio - Load (URL or Local)",
    "Load Audio Any Optional": "Audio - Load (URL or Local, Optional)",
    "Load Audio Any Stereo": "Audio - Load Stereo (URL or Local)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
