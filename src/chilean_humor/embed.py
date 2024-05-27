from openai import OpenAI
from pydantic import BaseModel
import datetime
from chilean_humor.config import CONFIG

client = OpenAI()

class JokeChunk(BaseModel):
    routine_id: int
    show_id: int
    event_name: str
    show_name: str
    start_timestamp: datetime.time
    text: str
    video_id: str

class EmbedJokeChunks:
    def __init__(self, model_name: str = CONFIG["embedding_model"]):
        self.embedding_model = model_name
    
    def __call__(self, chunk: JokeChunk):
        text = chunk.text
        text = text.replace("\n", " ")
        embedding = client.embeddings.create(input = [text], model=self.embedding_model).data[0].embedding
        t = chunk.start_timestamp
        start_time= (t.hour * 60 + t.minute) * 60 + t.second

        return {"text": text,
                "start_time": start_time,
                "routine_id": chunk.routine_id,
                "show_id": chunk.show_id,
                "event_name": chunk.event_name,
                "show_name": chunk.show_name,
                "url": f"https://www.youtube.com/watch?v={chunk.video_id}&start={start_time}",
                "embedding": embedding}