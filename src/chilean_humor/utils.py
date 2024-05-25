import re

def extract_video_id(url: str) -> str:
    match = re.search(
        r"^(?:https?:\/\/)?(?:www\.)?(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.+&v=))((\w|-){11})(?:\S+)?$",
        url,
    )
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid youtube url")