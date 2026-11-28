from functools import partial

from sidon_assets import asset_ready, download_assets


APOLLO_ASSETS = {
    "pytorch_model.bin": (66541845, "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76")
}
APOLLO_URL = "https://huggingface.co/JusperLee/Apollo"


def apollo_ready(folder):
    return all(asset_ready(folder / name, size) for name, (size, _) in APOLLO_ASSETS.items())


download_apollo = partial(download_assets, assets=APOLLO_ASSETS, base_url=f"{APOLLO_URL}/resolve/main")
