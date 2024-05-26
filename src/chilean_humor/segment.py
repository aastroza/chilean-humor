# Code from here: https://github.com/jxnl/youtubechapters-backend

from dataclasses import dataclass, field, asdict
from datetime import timedelta
from typing import List
import json

from loguru import logger
from chilean_humor.joke import create_jokes_from_transcript

@dataclass
class Segment:
    start_time: float
    end_time: float
    transcript: str = field(repr=False)
    transcript_length: int = field(init=False, default=0)
    timestamp: str = field(init=False, repr=True)
    from_whisper: bool = field(default=False)
    language: str = field(default="en")

    def __post_init__(self):
        self.transcript_length = len(self.transcript)
        self.start_time = round(self.start_time)
        self.timestamp = str(timedelta(seconds=self.start_time))

    def to_str(self, video_id):
        if len(self.transcript) > 0:
            return (
                "language:{lang} timestamp:{ts} url:{url}\ntranscript:{transcript}".format(
                    lang=self.language,
                    ts=self.timestamp,
                    url=f"https://youtu.be/{video_id}?t={self.start_time}s",
                    transcript=self.transcript,
                ).strip()
                + "\n"
            )
        else:
            return ""
    
    def to_prompt(self):
        if len(self.transcript) > 0:
            return (
                "language:{lang} timestamp:{ts} transcript:{transcript}".format(
                    lang=self.language,
                    ts=self.timestamp,
                    transcript=self.transcript,
                ).strip()
                + "\n"
            )
        else:
            return ""
    
    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)

    def from_json(json_line):
        dict_line = json.loads(json_line)
        return Segment(
            language=dict_line["language"],
            start_time=dict_line["start_time"],
            end_time=dict_line["end_time"],
            transcript=dict_line["transcript"],
        )

def group_speech_segments(
    segments: List[Segment], max_length=300
):
    phrases = []
    current_segment = segments[0]
    current_transcript = current_segment.transcript
    current_start_time = current_segment.start_time
    from_whisper = current_segment.from_whisper

    current_segment.transcript = current_segment.transcript.replace("[Música]", "")
    current_segment.transcript = current_segment.transcript.replace("[Aplausos]", "")

    for segment in segments:
        previous_segment = current_segment
        current_segment = segment

        current_segment.transcript = current_segment.transcript.replace("[Música]", "")
        current_segment.transcript = current_segment.transcript.replace("[Aplausos]", "")

        is_pause = (current_segment.start_time - previous_segment.end_time) > 0.1
        is_long = current_segment.start_time - current_start_time > 1
        is_too_long = len(current_transcript) > max_length

        if (is_long and is_pause) or is_too_long:
            phrases.append(
                Segment(
                    language=current_segment.language,
                    start_time=current_start_time,
                    end_time=previous_segment.end_time,
                    transcript=current_transcript.strip(),
                    from_whisper=from_whisper,
                )
            )
            current_transcript = current_segment.transcript
            current_start_time = current_segment.start_time
        else:
            current_transcript += " " + current_segment.transcript

    phrases.append(
                Segment(
                    language=current_segment.language,
                    start_time=current_start_time,
                    end_time=previous_segment.end_time,
                    transcript=current_transcript.strip(),
                    from_whisper=from_whisper,
                )
            )
    
    return phrases

def extract_jokes_from_segments(
        segments, chunk=300 * 10
):
    repertoires = []
    text = ""

    for block in segments:

        if len(text) < chunk:
           text += f"\n{block.to_prompt()}"
        else:
            logger.info("Extracting jokes.")
            repertoire = create_jokes_from_transcript(text)
            repertoires.append(repertoire)
            text = f"{block.to_prompt()}"
            logger.info(f"Extracted {len(repertoire.jokes)} jokes.")
        
    if text is not None and text != "":
        logger.info("Extracting jokes.")
        repertoire = create_jokes_from_transcript(text)
        repertoires.append(repertoire)
        logger.info(f"Extracted {len(repertoire.jokes)} jokes.")

    return repertoires    