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
vector_db = VectorDatabase(redis_client)
text_db = TextDatabase(redis_client)
code_db = CodeDatabase(redis_client)
pdf_db = PDFDatabase(redis_client)

class VectorDatabase:
    def __init__(self, redis_client):
        self.redis_client = r

    def add_item(self, key: str, value: list):
        value_str = ','.join(value)
        self.redis_client.set(key, value_str)

    def get_item(self, key: str) -> list:
        value_str = self.redis_client.get(key)
        if value_str is not None:
            return value_str.decode('utf-8').split(',')
        return None
        
    def add_vector(self, key: str, value: list):
        self.add_item(key, value)

    def get_vector(self, key: str) -> list:
        return self.get_item(key)

    def search_vectors(self, query_embedding, top_k=5):
        return self.search(query_embedding, top_k)

    def recommend_vectors(self, item_key, top_k=5):
        return self.recommend(item_key, top_k)

    def get_keys_by_prefix(self, prefix):
        return [key.decode('utf-8') for key in self.redis_client.scan_iter(f"{prefix}*")]

    def search(self, query_embedding, top_k=5):
        keys = self.get_keys_by_prefix('repo:') + self.get_keys_by_prefix('pdf:')
        embeddings = [np.array(self.get_item(key), dtype=float) for key in keys]
        similarities = cosine_similarity([query_embedding], embeddings)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(keys[i], similarities[i]) for i in top_indices]

    def recommend(self, item_key, top_k=5):
        item_embedding = np.array(self.get_item(item_key), dtype=float)
        return self.search(item_embedding, top_k)

class TextDatabase:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def add_item(self, key: str, value: str):
        self.redis_client.set(key, value)

    def get_item(self, key: str) -> str:
        value = self.redis_client.get(key)
        if value is not None:
            return value.decode('utf-8')
            
    def add_text(self, key: str, value: str):
        self.add_item(key, value)

    def get_text(self, key: str) -> str:
        return self.get_item(key)

    def search_texts(self, query: str) -> list:
        return self.search(query)
        
    def get_keys_by_prefix(self, prefix):
        return [key.decode('utf-8') for key in self.redis_client.scan_iter(f"{prefix}*")]
        
    def search(self, query: str, keys_prefix: str = "text:") -> list:
        keys = self.get_keys_by_prefix(keys_prefix)
        results = []
        for key in keys:
            item_value = self.get_item(key)
            if query.lower() in item_value.lower():
                results.append((key, item_value))
        return results
        
class CodeDatabase:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def add_item(self, key: str, value: str):
        self.redis_client.set(key, value)

    def get_item(self, key: str) -> str:
        value = self.redis_client.get(key)
        if value is not None:
            return value.decode('utf-8')   
   
    def add_code(self, key: str, value: str):
        self.add_item(key, value)

    def get_code(self, key: str) -> str:
        return self.get_item(key)

    def search_code(self, query: str) -> list:
        return self.search(query)

    def get_keys_by_prefix(self, prefix):
        return [key.decode('utf-8') for key in self.redis_client.scan_iter(f"{prefix}*")]

    def search(self, query: str, keys_prefix: str = "code:") -> list:
        keys = self.get_keys_by_prefix(keys_prefix)
        results = []
        for key in keys:
            item_value = self.get_item(key)
            if query.lower() in item_value.lower():
                results.append((key, item_value))
        return results

class PDFDatabase:
    def __init__(self, redis_client):
        self.redis_client = redis_client

    def add_item(self, key: str, value: bytes):
        self.redis_client.set(key, value)

    def get_item(self, key: str) -> bytes:
        return self.redis_client.get(key)

    def get_keys_by_prefix(self, prefix):
        return [key.decode('utf-8') for key in self.redis_client.scan_iter(f"{prefix}*")]

    def add_pdf(self, key: str, value: bytes):
        self.add_item(key, value)

    def get_pdf(self, key: str) -> bytes:
        return self.get_item(key)

    def search_pdfs(self, query: str) -> list:
        return self.search(query)