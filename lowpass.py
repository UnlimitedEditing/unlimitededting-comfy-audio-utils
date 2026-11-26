"""UE_AudioLowpass: FFT low-pass for an AUDIO. Removes everything above cutoff_hz (cosine taper over the 600 Hz below it).

Why: Graydient's MiniMax H3 soundtrack is band-limited (32 kHz audio, nothing real above ~16 kHz). A 44.1 kHz audio model
such as Stable Audio 3 re-synthesising it fills 16-20 kHz with invented hiss; cutting at the source's own limit hides that.
cutoff_hz <= 0 or >= Nyquist passes the audio through unchanged."""
import math
import torch


class AudioLowpass:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",),
                             "cutoff_hz": ("FLOAT", {"default": 15500.0, "min": 0.0, "max": 96000.0, "step": 10.0})}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "audio/utils"

    def run(self, audio, cutoff_hz):
        wav, sr = audio["waveform"], int(audio["sample_rate"])
        if cutoff_hz <= 0 or cutoff_hz >= sr / 2:
            return (audio,)
        n = wav.shape[-1]
        spec = torch.fft.rfft(wav.float(), dim=-1)
        freqs = torch.fft.rfftfreq(n, d=1.0 / sr).to(spec.device)
        taper = 600.0
        gain = torch.ones_like(freqs)
        lo = cutoff_hz - taper
        ramp = (freqs > lo) & (freqs < cutoff_hz)
        gain[ramp] = 0.5 * (1 + torch.cos(math.pi * (freqs[ramp] - lo) / taper))
        gain[freqs >= cutoff_hz] = 0.0
        out = torch.fft.irfft(spec * gain, n=n, dim=-1).to(wav.dtype)
        return ({**audio, "waveform": out, "sample_rate": sr},)


NODE_CLASS_MAPPINGS = {"UE_AudioLowpass": AudioLowpass}
NODE_DISPLAY_NAME_MAPPINGS = {"UE_AudioLowpass": "Audio - low-pass (FFT)"}
