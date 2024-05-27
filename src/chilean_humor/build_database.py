import sqlite_utils
import pandas as pd
import json
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

def convert_to_seconds(time_str):
    h, m, s = map(int, time_str.split(':'))
    return h * 3600 + m * 60 + s

# Create a connection to the database
db = sqlite_utils.Database("humor.db")

# Insert data from CSV files into the database
routines_df = pd.read_csv("data/routines.csv")
comedians_df = pd.read_csv("data/comedians.csv")
shows_df = pd.read_csv("data/shows.csv")
jokes_df = pd.read_csv("data/jokes.csv")

# Youtube transcripts

# read all .jsonl files in the transcripts folder
transcripts = []
for routine_id, url in zip(routines_df["ID"], routines_df["VIDEO"]):
    try:
        with open(f"transcripts/routine_{routine_id}_transcript.jsonl", "r", encoding="utf-8") as file:
            for line in file:
                transcript = json.loads(line)
                transcript["ID"] = routine_id
                video_id = extract_video_id(url)
                transcripts.append({"ID": routine_id, "TRANSCRIPT": transcript["transcript"], "TIMESTAMP": transcript["timestamp"], "URL": f"https://www.youtube.com/watch?v={video_id}&start={transcript["start_time"]}"})
    except FileNotFoundError:
        continue

transcripts_df = pd.DataFrame(transcripts)

jokes_df["START_TIME"] = jokes_df["start_timestamp"].apply(convert_to_seconds)
jokes_df["URL"] = jokes_df.apply(lambda row: f"https://www.youtube.com/watch?v={row['video_id']}&start={row['START_TIME']}", axis=1)
jokes_df["ID"] = range(1, len(jokes_df) + 1)
jokes_df = jokes_df[["ID", "routine_id", "show_id", "event_name", "start_timestamp", "text", "URL"]]
jokes_df.columns = ["ID", "ROUTINEID", "SHOWID", "EVENTNAME", "TIMESTAMP", "TEXT", "URL"]


db["routines"].insert_all(routines_df.to_dict(orient="records"), alter=True, pk="ID")
db["comedians"].insert_all(comedians_df.to_dict(orient="records"), alter=True, pk="ID")
db["shows"].insert_all(shows_df.to_dict(orient="records"), alter=True, pk="ID")
db["transcripts"].insert_all(transcripts_df.to_dict(orient="records"), alter=True, pk=("ID", "TIMESTAMP"))
db["jokes"].insert_all(jokes_df.to_dict(orient="records"), alter=True, pk=("ID"))

# Add foreign keys
db["comedians"].add_foreign_key("SHOWID", "shows", "ID")
db["routines"].add_foreign_key("SHOWID", "shows", "ID")
db["transcripts"].add_foreign_key("ID", "routines", "ID")
db["jokes"].add_foreign_key("ROUTINEID", "routines", "ID")
db["jokes"].add_foreign_key("SHOWID", "shows", "ID")