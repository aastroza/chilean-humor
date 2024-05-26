import pandas as pd
from loguru import logger
from typing import List

from chilean_humor.segment import extract_jokes_from_segments, Segment


def extract_repertories(
        segments: List[Segment],
    ):
    repertoires = extract_jokes_from_segments(segments = segments)
    return repertoires

def main():
    routine_id = 199
    filename = f"transcripts/routine_{routine_id}_transcript.jsonl"
    segments = []

    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            segment = Segment.from_json(line)
            segments.append(segment)
    repertoires = extract_repertories(segments = segments)

    for r in repertoires:
        for joke in r.jokes:
            try:
                if len(joke.corrected_transcript) > 0:
                    with open(f"jokes/routine_{routine_id}_repertoire.jsonl", "a", encoding="utf-8") as file:
                        json_line = joke.json()
                        file.write(json_line + "\n")
            except Exception as e:
                logger.info(f"Error writing joke for {routine_id} {e}")
                continue


if __name__ == "__main__":
    main()