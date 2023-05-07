import os
import glob
import openai
import shutil
import stat
import tempfile
import pdfplumber
from git import Repo
import redis
import asyncio
from data_structures import CONVERSATION_HISTORY_KEY
from gpt import gpt_interaction, SlidingWindowEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

redis_client = redis.Redis(host='localhost', port=6379, db=0)
vector_db = Database(redis_client, 'vector')
text_db = Database(redis_client, 'text')
code_db = Database(redis_client, 'code')
pdf_db = Database(redis_client, 'pdf')
chat_history_db = Database(redis_client, 'chat_history')
summarized_history_db = Database(redis_client, 'summarized_history')

class Database:
    def __init__(self, redis_client, data_type: str):
        self.redis_client = redis_client
        self.data_type = data_type

    def add_item(self, key: str, value):
        self.redis_client.set(f"{self.data_type}:{key}", value)
        
    def get_item(self, key: str):
        value = self.redis_client.get(f"{self.data_type}:{key}")
        if value is not None:
            if self.data_type == "vector":
                return list(map(float, value.decode('utf-8').split(',')))
            else:
                return value.decode('utf-8') if isinstance(value, bytes) else value
        return None

    def get_keys_by_prefix(self, prefix=None):
        if prefix is None:
            prefix = self.data_type
        with self.redis_client.pipeline() as pipe:
            for key in pipe.scan_iter(f"{prefix}:*", match=f"{prefix}:*"):
                yield key.decode('utf-8')

    def search(self, query: str, keys_prefix: str = None) -> list:
        if keys_prefix is None:
            keys_prefix = self.data_type
        keys = self.get_keys_by_prefix(keys_prefix)
        results = []
        for key in keys:
            item_value = self.get_item(key)
            if query.lower() in item_value.lower():
                results.append((key, item_value))
        return results

    def search_vectors(self, query_embedding, top_k=5):
        if self.data_type != "vector":
            raise ValueError("search_vectors method can only be used with a 'vector' Database instance")
        keys = self.get_keys_by_prefix()
        embeddings = [np.array(self.get_item(key), dtype=float) for key in keys]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(keys[i], similarities[i]) for i in top_indices]

    def recommend_vectors(self, item_key, top_k=5):
        if self.data_type != "vector":
            raise ValueError("recommend_vectors method can only be used with a 'vector' Database instance")
        item_embedding = np.array(self.get_item(item_key), dtype=float)
        return self.search_vectors(item_embedding, top_k)

    def get_all_items(self):
        keys = self.get_keys_by_prefix()
        return [(key, self.get_item(key)) for key in keys]

    def add_text(self, key: str, value: str):
        self.add_item(key, value)

    def get_text(self, key: str) -> str:
        return self.get_item(key)

    def add_code(self, key: str, value: str):
        self.add_item(key, value)

    def get_code(self, key: str) -> str:
        return self.get_item(key)

    def add_vector(self, key: str, value: list):
        value_str = ','.join(map(str, value))
        self.add_item(key, value_str)

    def get_vector(self, key: str) -> list:
        return self.get_item(key)

    def add_pdf(self, key: str, value: bytes):
        self.add_item(key, value)

    def get_pdf(self, key: str) -> bytes:
        return self.get_item(key)
