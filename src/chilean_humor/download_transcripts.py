import pandas as pd
import re
from loguru import logger
from chilean_humor.transcribe import transcribe_youtube
from chilean_humor.segment import group_speech_segments

def extract_video_id(url: str) -> str:
    match = re.search(
        r"^(?:https?:\/\/)?(?:www\.)?(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))((\w|-){11})(?:\S+)?$",
        url,
    )
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid youtube url")

def download_transcript(
        url: str,
    ):
    video_id = extract_video_id(url)
    phrases = transcribe_youtube(video_id)
    phrases = group_speech_segments(phrases, max_length=300)
    return phrases

def main():

    routines_df = pd.read_csv("data/routines.csv")

    # filter routines with no youtube video
    routines_df = routines_df[routines_df['VIDEO'].notnull()]

    for index, row in routines_df.iterrows():

        try:
            url = row['VIDEO']
            phrases = download_transcript(url)

            logger.info(f"Downloading transcript for {url}")
            with open(f"transcripts/routine_{row['ID']}_transcript.jsonl", "w", encoding="utf-8") as file:
                for phrase in phrases:
                    json_line = phrase.to_json()
                    file.write(json_line + "\n")
        except Exception as e:
            logger.info(f"Error downloading transcript for {url} {e}")

if __name__ == "__main__":
    main()