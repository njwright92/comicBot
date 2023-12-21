import csv
import re
import os
import json
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


def download_audio_and_get_title(video_id, ydl):
    audio_file_path = f'{audio_folder}/{video_id}.mp3'
    title = None
    info_dict = ydl.extract_info(
        f'https://www.youtube.com/watch?v={video_id}', download=False)
    title = sanitize_filename(info_dict.get('title', video_id))

    transcript_file = f"{transcripts_folder}/{title}.txt"
    # Check if either the audio file or the transcript file already exists
    if not os.path.exists(audio_file_path) and not os.path.exists(transcript_file):
        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])
    else:
        print(f"Skipping {video_id} as audio or transcript already exists.")
        title = None  # Set title to None to indicate skipping

    return title


def transcribe_audio_with_whisper(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    return result["text"]


def process_video(video_id, video_url, ydl):
    title = download_audio_and_get_title(video_id, ydl)
    if title:
        # Change the extension to .json
        transcript_file = f"{transcripts_folder}/{title}.json"
        audio_file = f"{audio_folder}/{video_id}.mp3"
        transcript_text = transcribe_audio_with_whisper(audio_file)

        # Include the video URL in the transcript data
        transcript_data = {
            "url": video_url,
            "title": title,
            "transcript": transcript_text
        }

        with open(transcript_file, 'w', encoding="utf-8") as outfile:
            json.dump(transcript_data, outfile, ensure_ascii=False,
                      indent=4)  # Write as formatted JSON
    else:
        print(f"Skipped processing for {video_id}.")


def worker(ydl):
    while True:
        video_id, video_url = video_queue.get()
        process_video(video_id, video_url, ydl)
        video_queue.task_done()


def download_transcripts_from_csv(file_path, ydl):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in tqdm(reader):
            url = row[0]
            if "list=" in url:
                try:
                    info_dict = ydl.extract_info(url, download=False)
                    if 'entries' in info_dict:
                        for video in info_dict['entries']:
                            if video:
                                video_url = video.get('webpage_url')
                                video_queue.put((video['id'], video_url))
                    else:
                        print(f"No entries found for playlist: {url}")
                except Exception as e:
                    print(f"Error extracting playlist information: {e}")
            else:
                video_id = url.split('v=')[-1]
                video_queue.put((video_id, url))


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
    for i in range(4):
        t = threading.Thread(target=worker, args=(ydl,))
        t.daemon = True
        t.start()

    csv_path = 'urls.csv'
    download_transcripts_from_csv(csv_path, ydl)

    video_queue.join()