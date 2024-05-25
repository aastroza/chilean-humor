import sqlite_utils
import pandas as pd

# Create a connection to the database
db = sqlite_utils.Database("humor.db")

# Insert data from CSV files into the database
routines_df = pd.read_csv("data/routines.csv")
comedians_df = pd.read_csv("data/comedians.csv")
shows_df = pd.read_csv("data/shows.csv")

db["routines"].insert_all(routines_df.to_dict(orient="records"), alter=True, pk="ID")
db["comedians"].insert_all(comedians_df.to_dict(orient="records"), alter=True, pk="ID")
db["shows"].insert_all(shows_df.to_dict(orient="records"), alter=True, pk="ID")

# Add foreign keys
db["comedians"].add_foreign_key("SHOWID", "shows", "ID")
db["routines"].add_foreign_key("SHOWID", "shows", "ID")