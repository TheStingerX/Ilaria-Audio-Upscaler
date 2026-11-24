"""Official Sidon INT8 assets, verified against Hugging Face LFS hashes."""
import hashlib
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen


ASSETS = {
    "sidon-predictor.onnx": (195258154, "ff50dc41f309903ac5dc071afb969a3f950b15e77f7e226222fd0011e76910c7"),
    "sidon-vocoder.onnx": (105055596, "b2598f7fae8913ee42a4ddedc4bb50ccd76f15546ec1f93520c19187f4a33644"),
}
BASE = "https://huggingface.co/soniqo/Sidon-ONNX/resolve/main/int8"


def asset_ready(path, size):
    return path.is_file() and path.stat().st_size == size


def sidon_ready(folder):
    return all(asset_ready(folder / name, size) for name, (size, _) in ASSETS.items())


def download_sidon(folder, progress, cancelled):
    folder.mkdir(parents=True, exist_ok=True)
    total = sum(size for size, _ in ASSETS.values())
    received = 0
    for name, (size, expected_hash) in ASSETS.items():
        destination = folder / name
        if asset_ready(destination, size):
            received += size
            continue
        if cancelled():
            return False
        digest = hashlib.sha256()
        temporary = NamedTemporaryFile(dir=folder, suffix=".part", delete=False)
        pending = Path(temporary.name)
        try:
            request = Request(f"{BASE}/{name}", headers={"User-Agent": "Ilaria Audio Upcaler"})
            with temporary, urlopen(request, timeout=30) as response:
                progress(received, total)
                while not cancelled():
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    temporary.write(block)
                    digest.update(block)
                    received += len(block)
                    progress(received, total)
            if cancelled():
                return False
            if pending.stat().st_size != size or digest.hexdigest() != expected_hash:
                raise ValueError(f"Downloaded file failed verification: {name}")
            pending.replace(destination)
        finally:
            pending.unlink(missing_ok=True)
    return True
