"""Download setup wheels and install them as separate, visible steps."""
import subprocess
import sys
from tempfile import TemporaryDirectory


def install(group, requirements, index_options):
    pip = [sys.executable, "-m", "pip", "--disable-pip-version-check"]
    with TemporaryDirectory(prefix="ilaria-setup-") as wheels:
        print(f"Download     [--------------------] {group}", flush=True)
        subprocess.run(
            [*pip, "download", "--progress-bar", "on", "--only-binary=:all:",
             "--dest", wheels, *index_options, *requirements],
            check=True,
        )
        print(f"Download     [####################] {group} complete", flush=True)
        print(f"Installation [--------------------] {group}", flush=True)
        subprocess.run(
            [*pip, "install", "--no-index", "--find-links", wheels, *requirements],
            check=True,
        )
        print(f"Installation [####################] {group} complete", flush=True)


def main():
    group = sys.argv[1]
    if group == "app":
        install("App and FFmpeg", ["-r", "App/requirements.txt"], [])
    else:
        install(
            f"PyTorch ({'CUDA' if group == 'cu128' else 'CPU'})",
            [f"torch==2.11.0+{group}", f"torchaudio==2.11.0+{group}"],
            ["--index-url", f"https://download.pytorch.org/whl/{group}"],
        )


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
