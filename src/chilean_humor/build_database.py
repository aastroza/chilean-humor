import sqlite_utils
import pandas as pd
import json
from chilean_humor.utils import extract_video_id

# Create a connection to the database
db = sqlite_utils.Database("humor.db")

# Insert data from CSV files into the database
routines_df = pd.read_csv("data/routines.csv")
comedians_df = pd.read_csv("data/comedians.csv")
shows_df = pd.read_csv("data/shows.csv")

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

db["routines"].insert_all(routines_df.to_dict(orient="records"), alter=True, pk="ID")
db["comedians"].insert_all(comedians_df.to_dict(orient="records"), alter=True, pk="ID")
db["shows"].insert_all(shows_df.to_dict(orient="records"), alter=True, pk="ID")
db["transcripts"].insert_all(transcripts_df.to_dict(orient="records"), alter=True, pk=("ID", "TIMESTAMP"))

# Add foreign keys
db["comedians"].add_foreign_key("SHOWID", "shows", "ID")
db["routines"].add_foreign_key("SHOWID", "shows", "ID")
db["transcripts"].add_foreign_key("ID", "routines", "ID")