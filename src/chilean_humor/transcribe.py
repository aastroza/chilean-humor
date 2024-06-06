# Code from here: https://github.com/jxnl/youtubechapters-backend

from chilean_humor.segment import Segment
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from loguru import logger
from openai import OpenAI
from chilean_humor.download import download_youtube_video

from dotenv import load_dotenv
load_dotenv()

client = OpenAI()

def transcribe_youtube(
    video_id: str
) -> List[Segment]:
    
    phrases = []

    # this function will try to get the transcript from youtube
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Get either 'es' or the first generated transcript
        language_code = None
        for t in transcript_list:
            if t.is_generated:
                language_code = t.language_code
                break

        logger.info(f"Transcript {video_id} language code: {language_code}")

        transcript = YouTubeTranscriptApi.get_transcript(
            video_id, ("es", language_code)
        )
        logger.info("Transcript found on youtube no need to download video")
        for t in transcript:
            phrases.append(
                Segment(
                    language=language_code or "en",
                    start_time=t["start"],
                    end_time=t["start"] + t["duration"],
                    transcript=t["text"],
                )
            )
    except Exception as e:
        logger.info(
            f"Video has transcripts disabled or not found {video_id} {e}"
        )
        logger.info("Downloading video to extract audio")
        file_name = download_youtube_video(f"https://www.youtube.com/watch?v={video_id}")

        audio_file = open(file_name, "rb")

        logger.info(f"Transcript {video_id} using whisper")

        transcript = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
            )

        for t in transcript.segments:
            phrases.append(
                Segment(
                    language=transcript.language,
                    start_time=t["start"],
                    end_time=t["end"],
                    transcript=t["text"],
                    from_whisper=True
                )
            )
    
    return phrases
