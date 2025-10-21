# Audio Clip Processing Guide

This guide explains how to process transcription segments and generate audio clips for ASR evaluation.

## Overview

The `process_clips.py` script:
1. Reads all `.jsonl` files from the `transcripts/` folder
2. Filters segments with 10+ words in the transcript
3. Downloads corresponding YouTube videos
4. Extracts audio clips for each selected segment
5. Saves clips to the `audios/` directory
6. Creates a metadata file (`clips_metadata.json`) mapping audio paths to transcript text

## Prerequisites

### System Dependencies

You need to install the following system tools:

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-pip
pip3 install yt-dlp
```

#### On macOS:
```bash
brew install ffmpeg
pip3 install yt-dlp
```

#### On other systems:
- **ffmpeg**: Download from https://ffmpeg.org/download.html
- **yt-dlp**: Install via pip: `pip install yt-dlp`

### Verify Installation

```bash
ffmpeg -version
yt-dlp --version
```

## Usage

### Process All Clips

To process all transcription segments and generate clips:

```bash
python3 process_clips.py
```

This will:
- Create a `videos/` directory with downloaded audio from YouTube
- Create an `audios/` directory with extracted clips
- Generate `clips_metadata.json` with clip metadata

### Process a Test Sample

For testing purposes, you can modify the script to process only a few routines first. See the "Testing" section below.

## Output Structure

### Generated Directories

```
chilean-humor/
├── videos/              # Downloaded video audio files
│   ├── routine_24.m4a
│   ├── routine_27.m4a
│   └── ...
├── audios/              # Extracted audio clips
│   ├── routine_24_clip_000001.mp3
│   ├── routine_24_clip_000002.mp3
│   └── ...
└── clips_metadata.json  # Metadata file
```

### Metadata Format

The `clips_metadata.json` file contains an array of objects with the following structure:

```json
[
  {
    "audio_path": "./audios/routine_24_clip_000001.mp3",
    "transcript": "Por favor, gracias a ella, de inmediato a continuación...",
    "routine_id": 24,
    "start_time": 0,
    "end_time": 25.0,
    "duration": 25.0
  },
  ...
]
```

## Configuration

You can modify the following parameters in `process_clips.py`:

- `MIN_WORDS`: Minimum number of words to include a segment (default: 10)
- `TRANSCRIPTS_DIR`: Directory containing transcript files (default: './transcripts')
- `ROUTINES_CSV`: Path to routines metadata CSV (default: './data/routines.csv')
- `VIDEOS_DIR`: Output directory for downloaded videos (default: './videos')
- `AUDIOS_DIR`: Output directory for audio clips (default: './audios')
- `METADATA_OUTPUT`: Output path for metadata file (default: './clips_metadata.json')

## Testing

To test the script on a small sample first, you can create a test version:

```bash
python3 process_clips_test.py
```

This will process only the first routine with a video URL to verify everything works correctly.

## Troubleshooting

### yt-dlp fails to download

- Ensure you have a stable internet connection
- Update yt-dlp: `pip install --upgrade yt-dlp`
- Some videos may be region-restricted or unavailable

### ffmpeg fails to extract clips

- Verify ffmpeg is installed: `ffmpeg -version`
- Check that the source audio file was downloaded correctly
- Ensure the start/end times are valid

### Out of disk space

The script will download full videos and generate many clips. Ensure you have sufficient disk space:
- Videos: ~100-500 MB per routine
- Clips: Variable, but could be several GB total

You can delete the `videos/` directory after processing to save space if needed.

## Performance

Processing all clips may take several hours depending on:
- Number of segments to process
- Internet connection speed
- CPU performance

The script logs progress every 100 clips.
