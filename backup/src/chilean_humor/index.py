import os
import psycopg2
from pgvector.psycopg2 import register_vector
from loguru import logger
from dotenv import load_dotenv

from chilean_humor.config import EMBEDDING_DIMENSIONS, CONFIG

# Load .env file
load_dotenv()

def store_pg_results(chunk):
    with psycopg2.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            #embedding_json = json.dumps(chunk['embedding'])
            cur.execute("INSERT INTO clips (routine_id, show_id, event_name, show_name, start_time, text, url, embedding) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)", (chunk['routine_id'], chunk['show_id'], chunk['event_name'], chunk['show_name'], chunk['start_time'], chunk['text'], chunk['url'], chunk['embedding']))


def set_index(embedded_chunks, embedding_model_name=CONFIG["embedding_model"]):
    logger.info("Setting index")

    with psycopg2.connect(os.environ["DB_CONNECTION_STRING"]) as conn:
        with conn.cursor() as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
            register_vector(conn)
            cur.execute("DROP TABLE clips;")
            cur.execute(f'CREATE TABLE clips (id serial primary key, routine_id int, show_id int, event_name text, show_name text, start_time float, "text" text not null, url text, embedding vector({EMBEDDING_DIMENSIONS[embedding_model_name]}));')

    for i, chunk in enumerate(embedded_chunks):
        logger.info(f'Storing chunk {i+1}/{len(embedded_chunks)}')
        store_pg_results(chunk)