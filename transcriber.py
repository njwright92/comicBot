import os
import json
import whisper
from tqdm import tqdm

audio_folder = 'audio_files'
transcripts_folder = 'transcripts2'


def transcribe_audio_with_whisper(audio_file):
    model = whisper.load_model("small")
    result = model.transcribe(audio_file)
    return result["text"]


def transcribe_audio_files():
    os.makedirs(transcripts_folder, exist_ok=True)

    for audio_file in tqdm(os.listdir(audio_folder)):
        if audio_file.endswith('.mp3'):
            full_audio_path = os.path.join(audio_folder, audio_file)
            transcript_text = transcribe_audio_with_whisper(full_audio_path)
            video_id = os.path.splitext(audio_file)[0]

            transcript_data = {
                "title": video_id,
                "transcript": transcript_text
            }

            transcript_file = os.path.join(
                transcripts_folder, f"{video_id}.json")
            with open(transcript_file, 'w', encoding="utf-8") as outfile:
                json.dump(transcript_data, outfile,
                          ensure_ascii=False, indent=4)


if __name__ == "__main__":
    transcribe_audio_files()
