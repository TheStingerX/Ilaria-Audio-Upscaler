from pathlib import Path
import shutil
import subprocess


DEFAULT_OUTPUT_FORMAT = "WAV (PCM 24-bit)"


OUTPUT_FORMATS = {
    DEFAULT_OUTPUT_FORMAT: (".wav", ()),
    "WAV (PCM 16-bit)": (".wav", ("-c:a", "pcm_s16le")),
    "FLAC": (".flac", ("-c:a", "flac")),
    "MP3": (".mp3", ("-c:a", "libmp3lame", "-q:a", "2")),
    "M4A (AAC)": (".m4a", ("-c:a", "aac", "-b:a", "192k")),
    "M4A (ALAC)": (".m4a", ("-c:a", "alac")),
    "AAC (ADTS)": (".aac", ("-c:a", "aac", "-b:a", "192k")),
    "OGG (Vorbis)": (".ogg", ("-c:a", "libvorbis", "-q:a", "5")),
    "Opus": (".opus", ("-c:a", "libopus", "-b:a", "128k")),
    "AIFF (PCM)": (".aiff", ("-c:a", "pcm_s16be")),
    "WavPack": (".wv", ("-c:a", "wavpack")),
    "WMA": (".wma", ("-c:a", "wmav2", "-b:a", "128k")),
}


def convert_audio(source, destination, encoder_options, cancelled):
    bundled = Path(__file__).with_name("ffmpeg.exe")
    executable = str(bundled) if bundled.is_file() else shutil.which("ffmpeg")
    if not executable:
        from imageio_ffmpeg import get_ffmpeg_exe

        executable = get_ffmpeg_exe()
    if cancelled():
        return False
    command = [
        executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
        "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
        *encoder_options, str(destination),
    ]
    with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        while True:
            try:
                _, stderr = process.communicate(timeout=0.15)
                break
            except subprocess.TimeoutExpired:
                if cancelled():
                    process.kill()
                    process.communicate()
                    return False
        if cancelled():
            return False
        if process.returncode:
            detail = stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(detail or "FFmpeg could not convert the audio.")
    return True
