"""Utility for building a dataset that can be used to evaluate ASR systems.

The pipeline is intentionally similar to :mod:`generate_transcripts`, however
instead of downloading every transcript (or falling back to Whisper), it only
collects the segments that already have a transcript available in Spanish on
YouTube.  Each collected record contains the original video URL together with
the transcripted text and the timestamps that bound the utterance.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
from loguru import logger
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from chilean_humor.utils import extract_video_id


@dataclass
class TranscriptSegment:
    """Light-weight container for the ASR evaluation dataset rows."""

    video_url: str
    text: str
    start_time: float
    end_time: float


def _normalise_text(text: str) -> str:
    """Normalise the transcript text.

    The YouTube transcript API keeps original newline characters; replacing
    them with spaces results in cleaner, single-line segments which are more
    convenient for CSV storage.
    """

    return " ".join(text.split())


def fetch_spanish_segments(video_url: str) -> List[TranscriptSegment]:
    """Retrieve the Spanish transcript segments for a given video URL.

    Only transcripts that are already available in Spanish are considered. If
    the video does not expose Spanish captions, an empty list is returned.
    """

    video_id = extract_video_id(video_url)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["es"])
    except (NoTranscriptFound, TranscriptsDisabled, CouldNotRetrieveTranscript) as error:
        logger.info(
            f"Skipping video without available Spanish transcript {video_id}: {error}"
        )
        return []

    segments: List[TranscriptSegment] = []
    for entry in transcript:
        text = _normalise_text(entry.get("text", "").strip())
        if not text:
            continue

        start = float(entry.get("start", 0.0))
        duration = float(entry.get("duration", 0.0))
        end = start + duration

        segments.append(
            TranscriptSegment(
                video_url=video_url,
                text=text,
                start_time=start,
                end_time=end,
            )
        )

    return segments


def generate_dataset(rows: Iterable[TranscriptSegment]) -> pd.DataFrame:
    """Convert an iterable of :class:`TranscriptSegment` into a dataframe."""

    data = [
        {
            "video_url": segment.video_url,
            "text": segment.text,
            "start_time": segment.start_time,
            "end_time": segment.end_time,
        }
        for segment in rows
    ]
    return pd.DataFrame(data, columns=["video_url", "text", "start_time", "end_time"])


def main(output_path: str = "data/asr_dataset.csv") -> None:
    routines_path = "data/routines.csv"
    logger.info(f"Loading routines metadata from {routines_path}")
    routines_df = pd.read_csv(routines_path)
    routines_df = routines_df[routines_df["VIDEO"].notnull()]

    logger.info("Collecting Spanish transcript segments")
    all_segments: List[TranscriptSegment] = []
    for url in routines_df["VIDEO"]:
        segments = fetch_spanish_segments(url)
        if segments:
            logger.info(
                f"Added Spanish transcript segments for {url}: {len(segments)}"
            )
            all_segments.extend(segments)
        else:
            logger.info(f"No Spanish transcript available for {url}")

    dataset = generate_dataset(all_segments)
    if dataset.empty:
        logger.warning("No Spanish transcript segments found; dataset is empty")
    else:
        logger.info(f"Saving ASR evaluation dataset with {len(dataset)} rows")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
