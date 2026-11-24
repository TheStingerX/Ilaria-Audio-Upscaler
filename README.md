# Ilaria Audio Upscaler

A Windows desktop app for speech bandwidth extension with **AP-BWE** and speech restoration with **Sidon INT8**. Conversion runs locally. Preview the result and use **Save result** to keep a copy.

## Setup

Install 64-bit [Python 3.10](https://www.python.org/downloads/), then double-click `Setup.cmd`. It installs the app dependencies and FFmpeg, builds `Ilaria Audio Upscaler.exe`, and opens the app. Setup shows overall progress and separate download and installation progress. After setup, open the EXE directly.

Setup enables CUDA automatically when a compatible NVIDIA GPU is available. It falls back to CPU if CUDA cannot be enabled.

## Use

1. Choose **Sidon** for speech restoration or **AP-BWE** for bandwidth extension. Select a model profile for AP-BWE.
2. Use **Download Sidon INT8** or **Download checkpoint** if the model is missing. Downloads are stored in `App/checkpoints`. Sidon INT8 needs approximately 286 MiB.
3. Add audio, choose an output format, and click **Enhance audio**.
4. Audition the original and the result. Use **Save result** to choose where to keep the converted file. Unsaved previews are temporary.

## Upstream projects

- [AP-BWE](https://github.com/yxlu-0102/AP-BWE) by Ye-Xin Lu, Yang Ai, Hui-Peng Du, and Zhen-Hua Ling. See `LICENSE-AP-BWE.txt` and `LICENSE-AP-BWE-WEIGHTS.txt` for the upstream notices.
- [Sidon-ONNX](https://huggingface.co/soniqo/Sidon-ONNX) by Soniqo, exported from [Sidon](https://github.com/sarulab-speech/Sidon). See the model card for model licensing and attribution.
- [Speech Core](https://github.com/soniqo/speech-core) documents the Sidon feature extraction used as a reference for this integration.

Model weights, audio files, Python environments, and built executables are excluded from this source package.



