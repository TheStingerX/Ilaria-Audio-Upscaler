"""Download official AP-BWE generator weights from the authors' public Drive."""
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import Request, urlopen
import zipfile


FILE_IDS = {
    "2kto16k": "1UnWlrf9WdUdEuXC2Y_Ht7XBObE8T7DP0",
    "4kto16k": "1Wox4dETh8SoJi6mQA_ctRfmCNaq9_Vd9",
    "8kto16k": "1e9CnVDA9oasnv349djtMCAiz4WmMAWFA",
    "8kto48k": "1UBLrQS1Ad48v4hsxDx9jDUmfGSkWj-Pz",
    "12kto48k": "1JWCFn52z7iSOGYWAn9O2H5mWO-51XGKJ",
    "16kto48k": "1HYkD_5ha9GMrjzbTiSFiO1uQQwxXET15",
    "24kto48k": "1iD5YBJrHibplnrZsD4zShG56v9gneURK",
}


def download_checkpoint(profile, destination, progress, cancelled):
    file_id = FILE_IDS[profile]
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    request = Request(url, headers={"User-Agent": "AP-BWE Studio"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return True

    temporary = NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False)
    pending = Path(temporary.name)
    try:
        with temporary, urlopen(request, timeout=20) as response:
            total = int(response.headers.get("Content-Length") or 0)
            received = 0
            progress(received, total)
            while not cancelled():
                block = response.read(256 * 1024)
                if not block:
                    break
                if received == 0 and not block.startswith(b"PK\x03\x04"):
                    raise ValueError("Google Drive did not return a PyTorch checkpoint.")
                temporary.write(block)
                received += len(block)
                progress(received, total)
        if cancelled():
            return False
        if total and total != received:
            raise ValueError("The checkpoint download was incomplete.")
        with zipfile.ZipFile(pending) as archive:
            if not any(name.endswith("/data.pkl") for name in archive.namelist()):
                raise ValueError("The download does not contain a PyTorch checkpoint.")
            if archive.testzip() is not None:
                raise ValueError("The checkpoint archive is damaged.")
        if cancelled():
            return False
        try:
            pending.rename(destination)
        except FileExistsError:
            # Another instance completed the same download first.
            pass
        return True
    finally:
        pending.unlink(missing_ok=True)
