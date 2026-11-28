from pathlib import Path
import json
import os
from tempfile import TemporaryDirectory

import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.functional import resample

from datasets.dataset import amp_pha_istft, amp_pha_stft
from env import AttrDict
from models.model import APNet_BWE_Model
from studio_formats import convert_audio


def load_model(config, checkpoint, device):
    h = AttrDict(json.loads(Path(config).read_text(encoding="utf-8")))
    if device == "Auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device.lower())
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this PyTorch installation. Select CPU or Auto.")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    model = APNet_BWE_Model(h)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["generator"])
    return model.eval().to(device), h, device


def read_audio(source, staging, cancelled):
    try:
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
    except sf.LibsndfileError:
        decoded = staging / "decoded.wav"
        if not convert_audio(source, decoded, ("-c:a", "pcm_f32le", "-rf64", "auto"), cancelled):
            return None
        audio, rate = sf.read(decoded, dtype="float32", always_2d=True)
    if not len(audio):
        raise ValueError("The audio file is empty.")
    return audio, rate


def finish_audio(wav, destination, encoder_options, progress, cancelled):
    if cancelled():
        return False
    pending = wav
    if encoder_options:
        pending = wav.with_name(f"encoded{destination.suffix}")
        if not convert_audio(wav, pending, encoder_options, cancelled):
            return False
    os.rename(pending, destination)
    progress(1.0)
    return True


@torch.inference_mode()
def enhance(source, destination, model, h, device, progress, cancelled, encoder_options=()):
    with TemporaryDirectory(dir=destination.parent) as staging:
        staging = Path(staging)
        decoded = read_audio(source, staging, cancelled)
        if decoded is None:
            return False
        audio, rate = decoded
        step = 4 * rate
        context = rate // 4
        wav = staging / "enhanced.wav"
        with sf.SoundFile(wav, mode="w", samplerate=h.hr_sampling_rate,
                          channels=audio.shape[1], subtype="PCM_24") as output:
            for start in range(0, len(audio), step):
                if cancelled():
                    return False
                end = min(start + step, len(audio))
                left = max(0, start - context)
                right = min(len(audio), end + context)
                chunk = torch.from_numpy(audio[left:right].T.copy()).to(device)
                chunk = resample(chunk, rate, h.lr_sampling_rate)
                chunk = resample(chunk, h.lr_sampling_rate, h.hr_sampling_rate)
                offset = round(start * h.hr_sampling_rate / rate) - round(left * h.hr_sampling_rate / rate)
                size = round(end * h.hr_sampling_rate / rate) - round(start * h.hr_sampling_rate / rate)
                length = max(chunk.shape[-1], offset + size, h.n_fft)
                length = ((length + h.hop_size - 1) // h.hop_size) * h.hop_size
                chunk = F.pad(chunk, (0, length - chunk.shape[-1]))
                results = []
                for channel in chunk.split(1):
                    amplitude, phase, _ = amp_pha_stft(channel, h.n_fft, h.hop_size, h.win_size)
                    amplitude, phase, _ = model(amplitude, phase)
                    reconstructed = amp_pha_istft(amplitude, phase, h.n_fft, h.hop_size, h.win_size)
                    results.append(reconstructed[:, offset:offset + size])
                output.write(torch.cat(results, dim=0).T.cpu().numpy())
                progress(0.95 * end / len(audio))
        return finish_audio(wav, destination, encoder_options, progress, cancelled)
