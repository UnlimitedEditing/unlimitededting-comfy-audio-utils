import os
import tempfile
import urllib.request

import torchaudio

try:
    import folder_paths
except ImportError:
    folder_paths = None


class LoadAudioAny:
    """
    Loads AUDIO from a single STRING that may be either:
      • a bare filename already staged in ComfyUI's input/ directory, or
      • a full http(s):// URL to download.

    Some Graydient submission paths resolve the same field to a pre-staged
    local filename; others resolve it to a remote URL. This node accepts
    either without the workflow needing to know in advance which one it'll get.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_source": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    def run(self, audio_source: str):
        value = (audio_source or "").strip()
        if not value:
            raise RuntimeError("LoadAudioAny: audio_source is empty.")

        if value.startswith("http://") or value.startswith("https://"):
            audio_path = self._download(value)
        else:
            audio_path = self._resolve_local(value)

        waveform, sample_rate = torchaudio.load(audio_path)
        audio = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
        return (audio,)

    def _resolve_local(self, filename: str) -> str:
        if folder_paths is not None:
            try:
                audio_path = folder_paths.get_annotated_filepath(filename)
                if audio_path and os.path.isfile(audio_path):
                    return audio_path
            except Exception:
                pass
            input_dir = folder_paths.get_input_directory() if folder_paths else None
            if input_dir:
                candidate = os.path.join(input_dir, filename)
                if os.path.isfile(candidate):
                    return candidate
        if os.path.isfile(filename):
            return filename
        raise RuntimeError(f"LoadAudioAny: local file not found: {filename!r}")

    def _download(self, url: str) -> str:
        suffix = os.path.splitext(url.split("?")[0])[1] or ".oga"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="load_audio_any_")
        os.close(fd)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-LoadAudioAny"})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp_path, "wb") as out:
                out.write(resp.read())
        except Exception as e:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise RuntimeError(f"LoadAudioAny: failed to download {url!r}: {e}")
        return tmp_path


NODE_CLASS_MAPPINGS = {
    "Load Audio Any": LoadAudioAny,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Load Audio Any": "Audio - Load (URL or Local)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
