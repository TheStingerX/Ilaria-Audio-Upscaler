from collections import defaultdict
from pathlib import Path
from typing import Any
from tempfile import TemporaryDirectory

import soundfile as sf
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
from torch.torch_version import TorchVersion
from torchaudio.functional import resample

from apollo_vendor.apollo import Apollo as ApolloModel
from apollo_vendor.chunking import chunk_starts, run_model
from studio_engine import finish_audio, read_audio


RATE = 44100


class Apollo:
    def __init__(self, folder, device):
        if device == "Auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device.lower())
        if self.device.type == "cpu":
            torch.set_num_threads(min(4, torch.get_num_threads()))
        metadata = [DictConfig, ListConfig, ContainerMetadata, Metadata, AnyNode,
                    TorchVersion, defaultdict, Any, int, dict, list]
        with torch.serialization.safe_globals(metadata):
            weights = torch.load(folder / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model = ApolloModel(sr=RATE, win=20, feature_dim=256, layer=6)
        model.load_state_dict(weights["state_dict"])
        self.model = model.eval().to(self.device)

    @torch.inference_mode()
    def enhance(self, source, destination, progress, cancelled, encoder_options):
        with TemporaryDirectory(dir=destination.parent) as staging:
            staging = Path(staging)
            decoded = read_audio(source, staging, cancelled)
            if decoded is None:
                return False
            audio, rate = decoded
            samples = resample(torch.from_numpy(audio.T.copy()).unsqueeze(0), rate, RATE)
            length = samples.shape[-1]
            samples = F.pad(samples, (0, max(0, RATE // 10 - length)))
            chunk_size, overlap = 3 * RATE, RATE // 2
            total = len(chunk_starts(samples.shape[-1], chunk_size, overlap))
            completed = 0

            def process(chunk):
                nonlocal completed
                if cancelled():
                    raise InterruptedError
                output = self.model(chunk)
                completed += 1
                progress(0.95 * completed / total)
                return output

            try:
                result = run_model(process, samples, self.device, chunk_samples=chunk_size,
                                   overlap_samples=overlap, chunk_pad_samples=RATE // 2)
            except InterruptedError:
                return False
            wav = staging / "enhanced.wav"
            sf.write(wav, result[0, :, :length].T.numpy(), RATE, subtype="PCM_24")
            return finish_audio(wav, destination, encoder_options, progress, cancelled)
