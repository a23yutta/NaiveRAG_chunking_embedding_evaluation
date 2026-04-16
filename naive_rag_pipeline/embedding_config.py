from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_embedding_model(name: str):

    if name == "openai-small":
        return OpenAIEmbedding(model="text-embedding-3-small")

    elif name == "openai-large":
        return OpenAIEmbedding(model="text-embedding-3-large")
    else:
        raise ValueError(f"Unknown embedding model: {name}")