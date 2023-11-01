import csv
import re
import os
import yt_dlp
import whisper
from tqdm import tqdm
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


# Folder to save audio and transcripts
audio_folder = 'audio_files'
transcripts_folder = 'transcripts'


def sanitize_filename(filename):
    """
    Removes invalid characters from filename and truncates it if it's too long.
    """
    s = re.sub(r'[\\/*?:"<>|]', '', filename)
    if len(s) > 200:
        s = s[:200]
    return s


def download_audio_with_yt_dlp(video_id):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3'
        }],
        'outtmpl': f'{audio_folder}/{video_id}.%(ext)s'
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])


def transcribe_audio_with_whisper(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result["text"]


def process_video(video_id):
    audio_file = f"{audio_folder}/{video_id}.mp3"

    # Download audio
    download_audio_with_yt_dlp(video_id)

    # Transcribe audio
    transcript = transcribe_audio_with_whisper(audio_file)

    # Save the transcript
    with open(f"{transcripts_folder}/{video_id}.txt", 'w', encoding="utf-8") as outfile:
        outfile.write(transcript)


def download_transcripts_from_csv(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            url = row[0]
            if "list=" in url:
                # It's a playlist, but we'll skip for simplicity
                print("Playlist detected. Skipping for now.")
            else:
                # It's a single video
                video_id = url.split('v=')[-1]
                process_video(video_id)


if __name__ == "__main__":
    # Ensure folders exist
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(transcripts_folder, exist_ok=True)

    csv_path = 'urls.csv'
    download_transcripts_from_csv(csv_path)
