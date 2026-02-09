# Mappings
EMBEDDING_DIMENSIONS = {
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}
MAX_CONTEXT_LENGTHS = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-4-turbo": 128000,
}

CONFIG = {
    "chat_model": "gpt-4o",
    "embedding_model": "text-embedding-3-small"
}