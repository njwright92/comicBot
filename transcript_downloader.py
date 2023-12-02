import csv
import re
import os
import yt_dlp
import whisper
from tqdm import tqdm
import ssl
import threading
from queue import Queue

ssl._create_default_https_context = ssl._create_unverified_context

# Folder to save audio and transcripts
audio_folder = 'audio_files'
transcripts_folder = 'transcripts2'

# Queue for processing videos
video_queue = Queue()


def sanitize_filename(filename):
    """
    Removes invalid characters from filename and truncates it if it's too long.
    """
    s = re.sub(r'[\\/*?:"<>|]', '', filename)
    if len(s) > 200:
        s = s[:200]
    return s


def download_audio_with_yt_dlp(video_id, ydl):
    audio_file_path = f'{audio_folder}/{video_id}.mp3'
    if not os.path.exists(audio_file_path):
        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
    else:
        print(f"Audio file for {video_id} already exists, skipping download.")


def transcribe_audio_with_whisper(audio_file):
    model = whisper.load_model("large")
    result = model.transcribe(audio_file)
    return result["text"]


def process_video(video_id, ydl):
    transcript_file = f"{transcripts_folder}/{video_id}.txt"
    # Check if transcript file already exists
    if not os.path.exists(transcript_file):
        audio_file = f"{audio_folder}/{video_id}.mp3"
        download_audio_with_yt_dlp(video_id, ydl)
        transcript = transcribe_audio_with_whisper(audio_file)
        with open(transcript_file, 'w', encoding="utf-8") as outfile:
            outfile.write(transcript)
    else:
        print(f"Transcript for {video_id} already exists, skipping.")


def worker(ydl):
    while True:
        video_id = video_queue.get()
        process_video(video_id, ydl)
        video_queue.task_done()


def download_transcripts_from_csv(file_path, ydl):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            url = row[0]
            if "list=" in url:
                info_dict = ydl.extract_info(url, download=False)
                for video in info_dict['entries']:
                    if video:
                        video_queue.put(video['id'])
            else:
                video_id = url.split('v=')[-1]
                video_queue.put(video_id)


if __name__ == "__main__":
    os.makedirs(audio_folder, exist_ok=True)
    os.makedirs(transcripts_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{audio_folder}/%(id)s.%(ext)s',
        'ignoreerrors': True,
        'extract_flat': True,
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)

    # Create and start worker threads
    for i in range(4):  # Adjust the number of threads as needed
        t = threading.Thread(target=worker, args=(ydl,))
        t.daemon = True
        t.start()

    csv_path = 'urls.csv'
    download_transcripts_from_csv(csv_path, ydl)

    video_queue.join()
