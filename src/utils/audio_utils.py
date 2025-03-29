import subprocess
import os

def has_audio_track(mp4_path):
    """
    Kiểm tra file MP4 có chứa luồng âm thanh hay không
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a:0',
        '-show_entries', 'stream=codec_type',
        '-of', 'csv=p=0',
        mp4_path
    ]
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode().strip()
        return bool(output)
    except subprocess.CalledProcessError as e:
        print(f"ffprobe error for {mp4_path}: {e.output.decode()}")
        return False

def convert_mp4_to_wav(mp4_path, wav_path, sample_rate=48000, channels=2):
    """
    Chuyển đổi file MP4 sang WAV sử dụng ffmpeg.
    """
    command = [
        'ffmpeg',
        '-y',  # Ghi đè file đích nếu tồn tại
        '-i', mp4_path,
        '-vn',  # Loại bỏ video
        '-acodec', 'pcm_s16le',
        '-ar', str(sample_rate),
        '-ac', str(channels),
        wav_path
    ]
    try:
        subprocess.check_output(command, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mp4_path} to WAV: {e.output.decode()}")
        return False
