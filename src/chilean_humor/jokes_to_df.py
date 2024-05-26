import glob
import json
import pandas as pd
from chilean_humor.embed import JokeChunk
from chilean_humor.utils import extract_number, extract_video_id

jokes_files = glob.glob("jokes/*.jsonl")

routines_df = pd.read_csv("data/routines.csv")
comedians_df = pd.read_csv("data/comedians.csv")
shows_df = pd.read_csv("data/shows.csv")

jokes_chunks = []

for f in jokes_files:
    routine_id = extract_number(f)
    with open(f, "r", encoding="utf-8") as file:
        for line in file:
            joke = json.loads(line)

            text = joke["corrected_transcript"]
            start_timestamp = joke["start_timestamp"]
            show_id = routines_df[routines_df["ID"] == routine_id]["SHOWID"].values[0]
            event_name = routines_df[routines_df["ID"] == routine_id]["EVENT"].values[0] + " " + str(routines_df[routines_df["ID"] == routine_id]["YEAR"].values[0])
            show_name = shows_df[shows_df["ID"] == show_id]["TITLE"].values[0]
            video_id = extract_video_id(routines_df[routines_df["ID"] == routine_id]["VIDEO"].values[0])

            jokes_chunks.append(JokeChunk(text=text,
                                            start_timestamp=start_timestamp,
                                            routine_id=routine_id,
                                            show_id=show_id,
                                            event_name=event_name,
                                            show_name=show_name,
                                            video_id=video_id).model_dump()
                                )

df = pd.DataFrame(jokes_chunks)
#df = df.stack().str.replace('\n', '\\n', regex=True).unstack()
df.to_csv("data/jokes.csv", index=False)

