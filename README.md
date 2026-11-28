# Ilaria Audio Upscaler

A Windows desktop app for speech bandwidth extension with **AP-BWE**, speech restoration with **Sidon INT8**, and music restoration with **Apollo**. Conversion runs locally. Preview the result and use **Save result** to keep a copy.

## Demo

https://github.com/user-attachments/assets/da4619a1-d846-417b-864a-b01607e8b609


## Setup

Install 64-bit [Python 3.10](https://www.python.org/downloads/), then double-click `Setup.cmd`. It installs the app dependencies and FFmpeg, builds `Ilaria Audio Upscaler.exe`, and opens the app. Setup shows overall progress and separate download and installation progress. After setup, open the EXE directly.

Setup enables CUDA automatically when a compatible NVIDIA GPU is available. It falls back to CPU if CUDA cannot be enabled.

## Use

1. Choose **Sidon** for speech restoration with denoising, **AP-BWE** for speech bandwidth extension, or **Apollo** for music degraded by lossy compression. Select a model profile for AP-BWE.
2. Use the model download button if the weights are missing. Downloads are stored in `App/checkpoints`. Sidon INT8 needs approximately 286 MiB.
3. Add audio, choose an output format (WAV defaults to 24-bit PCM, with 16-bit PCM also available), and click **Enhance audio**.
4. Audition the original and the result. Use **Save result** to choose where to keep the converted file. Unsaved previews are temporary.

## Recent changes

- **Apollo** restores compressed music at 44.1 kHz with CPU or NVIDIA CUDA support.
- **WAV export** supports PCM 24-bit (default) and PCM 16-bit.
- The player includes play, pause, stop, click-to-seek, dragging, and rates from 0.25× to 32×.

## Upstream projects

- [AP-BWE](https://github.com/yxlu-0102/AP-BWE) by Ye-Xin Lu, Yang Ai, Hui-Peng Du, and Zhen-Hua Ling. See `LICENSE-AP-BWE.txt` and `LICENSE-AP-BWE-WEIGHTS.txt` for the upstream notices.
- [Sidon-ONNX](https://huggingface.co/soniqo/Sidon-ONNX) by Soniqo, exported from [Sidon](https://github.com/sarulab-speech/Sidon). See the model card for model licensing and attribution.
- [Speech Core](https://github.com/soniqo/speech-core) documents the Sidon feature extraction used as a reference for this integration.

- [Apollo](https://github.com/JusperLee/Apollo) by Kai Li and Yi Luo. The vendored architecture and adapted chunking are under CC BY-SA 4.0. See `App/apollo_vendor/LICENSE` and `App/apollo_vendor/UPSTREAM.txt`.

The repository excludes model weights, audio files, Python environments, and built executables. The Windows ZIP includes the launcher and setup. Setup installs the runtime locally, and the app downloads each model on demand.
