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
    r'(voice|photo|video_note|video|audio|document|animation|sticker)file(\d+)\.([a-z0-9]+)$',
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


def _decode_audio_via_ffmpeg(path, sample_rate=32000):
    """Decode to a mono waveform tensor via ffmpeg subprocess (not torchaudio.load,
    see module docstring). Raw PCM output, not WAV, since ffmpeg's WAV header on a
    pipe has a placeholder frame count."""
    import numpy as np
    import torch

    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "s16le", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to decode {path}: {proc.stderr.decode(errors='replace')}")

    arr = np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0
    waveform = torch.from_numpy(arr.copy()).unsqueeze(0)  # [channels=1, samples]
    return waveform, sample_rate


def _resolve_reference_audio(value, source_label):
    value = (value or "").strip()
    if not value:
        return None

    if value.startswith("http://") or value.startswith("https://"):
        print(f"[LoadAudioAny] {source_label}={value!r} -> treating as URL")
        suffix = os.path.splitext(value.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(value, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path)

    reconstructed = _unmangle_telegram_file_url(value)
    if reconstructed is not None:
        print(f"[LoadAudioAny] {source_label}={value!r} -> reconstructed mangled Telegram URL {reconstructed!r}")
        suffix = os.path.splitext(reconstructed.split("?")[0])[1] or ".audio"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        _download(reconstructed, tmp_path)
        return _decode_audio_via_ffmpeg(tmp_path)

    candidates = [value]
    if folder_paths is not None:
        try:
            candidates.append(os.path.join(folder_paths.get_input_directory(), value))
        except Exception:
            pass

    for candidate in candidates:
        if os.path.isfile(candidate):
            print(f"[LoadAudioAny] {source_label}={value!r} -> found local file {candidate!r}")
            return _decode_audio_via_ffmpeg(candidate)

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


NODE_CLASS_MAPPINGS = {
    "Load Audio Any": LoadAudioAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Audio Any": "Audio - Load (URL or Local)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
