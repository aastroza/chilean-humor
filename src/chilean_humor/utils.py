import re
import subprocess
import os

def extract_video_id(url: str) -> str:
    match = re.search(
        r"^(?:https?:\/\/)?(?:www\.)?(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))((\w|-){11})(?:\S+)?$",
        url,
    )
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid youtube url")

def extract_number(s):
    match = re.search(r'\d+', s)
    if match:
        return int(match.group())
    else:
        return None

def execute_bash(command):
    results = subprocess.run(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return results

def extract_routines_ids(folder='jokes'):
    # Initialize an empty list to store the routine_ids
    routine_ids = []

    # Regular expression pattern to match the routine_id in the filename
    pattern = re.compile(r'routine_(\d+)_repertoire\.jsonl')

    # Iterate over the files in the directory
    for filename in os.listdir(folder):
        # Match the filename against the pattern
        match = pattern.match(filename)
        if match:
            # Extract the routine_id and append to the list
            routine_ids.append(int(match.group(1)))
    
    return routine_ids