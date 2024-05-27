import pandas as pd
from loguru import logger
from chilean_humor.embed import EmbedJokeChunks, JokeChunk
from chilean_humor.index import set_index
from chilean_humor.config import CONFIG

jokes = pd.read_csv("data/jokes.csv")

chunks = [JokeChunk(**joke) for joke in jokes.to_dict(orient="records")]

embedded_chunks = []
embedder = EmbedJokeChunks(CONFIG["embedding_model"])
for i, chunk in enumerate(chunks):
    logger.info(f"Embedding chunk {i+1}/{len(chunks)}")
    embedded_chunks.append(embedder(chunk))

set_index(embedded_chunks)