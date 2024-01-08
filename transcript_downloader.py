import csv
import re
import os
import yt_dlp
from tqdm import tqdm
import ssl
import threading
from queue import Queue

ssl._create_default_https_context = ssl._create_unverified_context

audio_folder = 'audio_files'  # Folder to save audio files

video_queue = Queue()


def sanitize_filename(filename):
    s = re.sub(r'[\\/*?:"<>|]', '', filename)
    return s[:200] if len(s) > 200 else s


def download_audio(video_id, ydl):
    audio_file_path = f'{audio_folder}/{video_id}.mp3'
    try:
        info_dict = ydl.extract_info(
            f'https://www.youtube.com/watch?v={video_id}', download=False)
        title = sanitize_filename(info_dict.get('title', video_id))

        if not os.path.exists(audio_file_path):
            ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
        else:
            print(f"Audio for {video_id} already exists.")
    except Exception as e:
        print(f"Error with video {video_id}: {e}")


def worker(ydl):
    while True:
        video_id = video_queue.get()
        download_audio(video_id, ydl)
        video_queue.task_done()


def download_audio_from_csv(file_path, ydl):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            video_id = row[0].split('v=')[-1]
            video_queue.put(video_id)


if __name__ == "__main__":
    os.makedirs(audio_folder, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{audio_folder}/%(id)s.%(ext)s',
        'ignoreerrors': True,
    }

    ydl = yt_dlp.YoutubeDL(ydl_opts)

    for i in range(4):
        t = threading.Thread(target=worker, args=(ydl,))
        t.daemon = True
        t.start()

    csv_path = 'urls.csv'
    download_audio_from_csv(csv_path, ydl)

    video_queue.join()
