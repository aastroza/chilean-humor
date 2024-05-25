# Code from here: https://github.com/jxnl/youtubechapters-backend

from chilean_humor.segment import Segment
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from loguru import logger

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
    
    return phrases
