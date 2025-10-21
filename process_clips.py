#!/usr/bin/env python3
"""
Script to process transcription segments and generate audio clips.

This script:
1. Reads all .jsonl files from the transcripts folder
2. Filters segments with 10+ words in the transcript
3. Downloads corresponding YouTube videos
4. Extracts audio clips for each selected segment
5. Saves clips to the /audios directory
6. Creates a metadata file mapping audio paths to transcript text
"""

import json
import os
import csv
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def load_routines_metadata(csv_path: str) -> Dict[int, str]:
    """
    Load routines metadata from CSV file.

    Returns:
        Dictionary mapping routine ID to YouTube URL
    """
    routines = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            routine_id = int(row['ID'])
            video_url = row['VIDEO'].strip()
            if video_url:  # Only include routines with valid video URLs
                routines[routine_id] = video_url

    logger.info(f"Loaded {len(routines)} routines with video URLs")
    return routines


def load_transcripts(transcripts_dir: str) -> Dict[int, List[dict]]:
    """
    Load all transcript files from the transcripts directory.

    Returns:
        Dictionary mapping routine ID to list of transcript segments
    """
    transcripts = {}
    transcripts_path = Path(transcripts_dir)

    for jsonl_file in transcripts_path.glob('routine_*_transcript.jsonl'):
        # Extract routine ID from filename
        match = re.search(r'routine_(\d+)_transcript\.jsonl', jsonl_file.name)
        if not match:
            continue

        routine_id = int(match.group(1))
        segments = []

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                segment = json.loads(line.strip())
                segments.append(segment)

        transcripts[routine_id] = segments
        logger.info(f"Loaded routine {routine_id} with {len(segments)} segments")

    return transcripts


def filter_segments(transcripts: Dict[int, List[dict]], min_words: int = 10) -> List[Tuple[int, dict]]:
    """
    Filter segments that have at least min_words words.

    Returns:
        List of tuples (routine_id, segment)
    """
    filtered = []

    for routine_id, segments in transcripts.items():
        for segment in segments:
            transcript_text = segment['transcript']
            word_count = count_words(transcript_text)

            if word_count >= min_words:
                filtered.append((routine_id, segment))

    logger.info(f"Filtered {len(filtered)} segments with {min_words}+ words")
    return filtered


def download_video(video_url: str, output_dir: str, routine_id: int) -> str:
    """
    Download YouTube video audio using yt-dlp.

    Returns:
        Path to downloaded audio file
    """
    output_template = os.path.join(output_dir, f'routine_{routine_id}.%(ext)s')

    # Check if already downloaded
    for ext in ['m4a', 'webm', 'mp3', 'opus']:
        existing_file = os.path.join(output_dir, f'routine_{routine_id}.{ext}')
        if os.path.exists(existing_file):
            logger.info(f"Video already downloaded: {existing_file}")
            return existing_file

    cmd = [
        'yt-dlp',
        '-f', 'bestaudio',
        '-o', output_template,
        '--no-playlist',
        video_url
    ]

    logger.info(f"Downloading video for routine {routine_id}: {video_url}")

    try:
        subprocess.run(cmd, check=True, capture_output=True)

        # Find the downloaded file
        for ext in ['m4a', 'webm', 'mp3', 'opus']:
            audio_file = os.path.join(output_dir, f'routine_{routine_id}.{ext}')
            if os.path.exists(audio_file):
                return audio_file

        raise FileNotFoundError(f"Could not find downloaded audio file for routine {routine_id}")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download video for routine {routine_id}: {e}")
        logger.error(f"stderr: {e.stderr.decode() if e.stderr else 'N/A'}")
        raise


def extract_audio_clip(
    source_audio: str,
    start_time: float,
    end_time: float,
    output_path: str
) -> None:
    """
    Extract audio clip from source audio file using ffmpeg.
    """
    duration = end_time - start_time

    cmd = [
        'ffmpeg',
        '-i', source_audio,
        '-ss', str(start_time),
        '-t', str(duration),
        '-acodec', 'libmp3lame',
        '-ar', '16000',  # 16kHz sample rate (common for ASR)
        '-ac', '1',      # Mono audio
        '-y',            # Overwrite output file
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract clip to {output_path}: {e}")
        raise


def process_clips(
    filtered_segments: List[Tuple[int, dict]],
    routines_metadata: Dict[int, str],
    videos_dir: str,
    audios_dir: str,
    metadata_output: str
) -> None:
    """
    Process all filtered segments and generate audio clips.
    """
    # Create directories
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(audios_dir, exist_ok=True)

    # Track which videos we've already downloaded
    downloaded_videos = {}

    # Prepare metadata file
    metadata = []

    total_clips = len(filtered_segments)
    logger.info(f"Processing {total_clips} clips...")

    for idx, (routine_id, segment) in enumerate(filtered_segments, 1):
        # Check if this routine has a video URL
        if routine_id not in routines_metadata:
            logger.warning(f"Routine {routine_id} has no video URL, skipping segment")
            continue

        # Download video if not already downloaded
        if routine_id not in downloaded_videos:
            try:
                video_url = routines_metadata[routine_id]
                source_audio = download_video(video_url, videos_dir, routine_id)
                downloaded_videos[routine_id] = source_audio
            except Exception as e:
                logger.error(f"Failed to download routine {routine_id}: {e}")
                continue

        source_audio = downloaded_videos[routine_id]

        # Generate output filename
        clip_filename = f"routine_{routine_id}_clip_{idx:06d}.mp3"
        clip_path = os.path.join(audios_dir, clip_filename)

        # Extract audio clip
        try:
            extract_audio_clip(
                source_audio,
                segment['start_time'],
                segment['end_time'],
                clip_path
            )

            # Add to metadata
            metadata.append({
                'audio_path': clip_path,
                'transcript': segment['transcript'],
                'routine_id': routine_id,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'duration': segment['end_time'] - segment['start_time']
            })

            if idx % 100 == 0:
                logger.info(f"Processed {idx}/{total_clips} clips")

        except Exception as e:
            logger.error(f"Failed to extract clip {idx}: {e}")
            continue

    # Save metadata
    logger.info(f"Saving metadata to {metadata_output}")
    with open(metadata_output, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Successfully processed {len(metadata)} clips")


def main():
    """Main execution function."""
    # Configuration
    TRANSCRIPTS_DIR = './transcripts'
    ROUTINES_CSV = './data/routines.csv'
    VIDEOS_DIR = './videos'
    AUDIOS_DIR = './audios'
    METADATA_OUTPUT = './clips_metadata.json'
    MIN_WORDS = 10

    logger.info("Starting clip processing...")

    # Load data
    logger.info("Loading routines metadata...")
    routines_metadata = load_routines_metadata(ROUTINES_CSV)

    logger.info("Loading transcripts...")
    transcripts = load_transcripts(TRANSCRIPTS_DIR)

    # Filter segments
    logger.info(f"Filtering segments with {MIN_WORDS}+ words...")
    filtered_segments = filter_segments(transcripts, min_words=MIN_WORDS)

    # Process clips
    process_clips(
        filtered_segments,
        routines_metadata,
        VIDEOS_DIR,
        AUDIOS_DIR,
        METADATA_OUTPUT
    )

    logger.info("Done!")


if __name__ == '__main__':
    main()
