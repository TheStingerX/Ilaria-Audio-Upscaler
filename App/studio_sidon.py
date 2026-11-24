"""Sidon speech restoration with the Kaldi/SeamlessM4T signal frontend."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from torchaudio.compliance.kaldi import fbank
from torchaudio.functional import resample

from studio_engine import finish_audio, read_audio
from studio_formats import convert_audio


INPUT_RATE = 16000
OUTPUT_RATE = 48000

# afftdn has 25 ms of latency at 48 kHz. Pad and trim to retain the timing.
DENOISE_FILTER = (
    "apad=pad_len=1200,afftdn=nr=8:nf=-46:gs=4,"
    "atrim=start_sample=1200,asetpts=PTS-STARTPTS"
)


class Sidon:
    def __init__(self, folder, device):
        settings = ort.SessionOptions()
        settings.log_severity_level = 3
        settings.intra_op_num_threads = min(4, torch.get_num_threads())
        providers = ["CPUExecutionProvider"]
        self.device = "cpu"
        if device != "CPU" and torch.cuda.is_available():
            index = int(device.split(":")[1]) if ":" in device else 0
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": index}))
            self.device = f"cuda:{index}"

        self.predictor = ort.InferenceSession(str(folder / "sidon-predictor.onnx"), settings,
                                              providers=providers)
        self.vocoder = ort.InferenceSession(str(folder / "sidon-vocoder.onnx"), settings,
                                            providers=providers)
        if self.device != "cpu" and any("CUDAExecutionProvider" not in session.get_providers()
                                        for session in (self.predictor, self.vocoder)):
            if device != "Auto":
                raise RuntimeError("ONNX Runtime could not initialize CUDA for Sidon. Select CPU.")
            self.device = "cpu"
            for session in (self.predictor, self.vocoder):
                session.set_providers(["CPUExecutionProvider"])

    def infer(self, samples):
        # Two right-context frames compensate for snip-edges in the filterbank.
        waveform = torch.nn.functional.pad(samples[None, :] * 32768.0, (0, 640))
        mel = fbank(waveform, sample_frequency=INPUT_RATE, num_mel_bins=80,
                    low_freq=20, high_freq=0, dither=0, window_type="povey")
        mel = (mel - mel.mean(dim=0)) / (mel.var(dim=0, unbiased=True) + 1e-5).sqrt()
        features = mel[: (len(mel) // 2) * 2].reshape(1, -1, 160).numpy()
        hidden = self.predictor.run(None, {"input_features": features})[0]
        return self.vocoder.run(None, {"features": hidden})[0].reshape(-1)

    @torch.inference_mode()
    def enhance(self, source, destination, progress, cancelled, encoder_options):
        with TemporaryDirectory(dir=destination.parent) as staging:
            staging = Path(staging)
            decoded = read_audio(source, staging, cancelled)
            if decoded is None:
                return False
            audio, rate = decoded
            duration = max(1, round(len(audio) * OUTPUT_RATE / rate))
            samples = resample(torch.from_numpy(audio.T.copy()), rate, INPUT_RATE)
            step = INPUT_RATE * 6
            context = INPUT_RATE // 2
            wav = staging / "enhanced.wav"
            with sf.SoundFile(wav, mode="w", samplerate=OUTPUT_RATE,
                              channels=audio.shape[1], subtype="PCM_16") as output:
                for start in range(0, samples.shape[-1], step):
                    end = min(start + step, samples.shape[-1])
                    left = max(0, start - context)
                    right = min(end + context, samples.shape[-1])
                    offset = (start - left) * 3
                    length = min(end * 3, duration) - start * 3
                    if length <= 0:
                        break
                    channels = []
                    for channel in samples[:, left:right]:
                        if cancelled():
                            return False
                        restored = self.infer(channel)[offset:offset + length]
                        channels.append(np.pad(restored, (0, length - len(restored))))
                    output.write(np.stack(channels, axis=1))
                    progress(0.95 * end / samples.shape[-1])
            cleaned = staging / "denoised.wav"
            if not convert_audio(wav, cleaned, ("-af", DENOISE_FILTER, "-c:a", "pcm_s16le"), cancelled):
                return False
            return finish_audio(cleaned, destination, encoder_options, progress, cancelled)


