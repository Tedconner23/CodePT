import os
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
from datastructures import CONVERSATION_HISTORY_KEY
from gpt import gpt_interaction, SlidingWindowEncoder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

r = redis.Redis(host='localhost', port=6379, db=0)

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

vector_db = VectorDatabase(redis_client=r)

async def get_ada_embeddings(text, model='text-embedding-ada-002'):
    loop = asyncio.get_event_loop()
    window_size = 4096
    step_size = 2048
    sliding_window_encoder = SlidingWindowEncoder(window_size, step_size)
    windows = sliding_window_encoder.encode(text)

    async def get_embeddings(window):
        response = await loop.run_in_executor(None, lambda: openai.Completion.create(engine=model, prompt=window, max_tokens=1, n=1, stop=None, temperature=.5))
        return response.choices[0].text.strip()

    embeddings = await asyncio.gather(*(get_embeddings(window) for window in windows))
    return embeddings

def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)

async def ingest_git_repo(repo_url: str, file_types: list = ['.cs', '.html', '.js', '.py']):
    print("Ingesting Git repository...")
    tmp_dir = tempfile.mkdtemp()
    repo = Repo.clone_from(repo_url, tmp_dir)

    async def process_file(file_path):
        print(f"Processing file: {file_path}")
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        analysis = await get_ada_embeddings(content)
        if analysis:
            vector_db.add_item(f"repo:{os.path.relpath(file_path, tmp_dir)}", analysis)
        else:
            print(f"No analysis data for {file_path}")

    tasks = [asyncio.ensure_future(process_file(file_path)) for file_path in glob.glob(tmp_dir + '/**', recursive=True) if os.path.isfile(file_path) and os.path.splitext(file_path)[-1] in file_types]
    await asyncio.gather(*tasks)
    shutil.rmtree(tmp_dir, onerror=remove_readonly)
    print("Ingestion complete.")

async def ingest_pdf_files(directory: str):
    print("Ingesting PDF files...")
    pdf_files = glob.glob(os.path.join(directory, '*.pdf'))

    async def process_pdf(pdf_file):
        print(f"Processing PDF: {pdf_file}")
        with pdfplumber.open(pdf_file) as pdf:
            content = ''.join(page.extract_text() for page in pdf.pages)
        analysis = await get_ada_embeddings(content)
        if analysis:
            vector_db.add_item(f"pdf:{os.path.basename(pdf_file)}", analysis)
        else:
            print(f"No analysis data for {pdf_file}")

    await asyncio.gather(*(process_pdf(pdf_file) for pdf_file in pdf_files))
    print("PDF ingestion complete.")

async def get_pdf_library() -> str:
    pdf_keys = vector_db.get_keys_by_prefix('pdf:')
    pdf_library = ''
    for key in pdf_keys:
        pdf_name = key.split('pdf:')[-1]
        pdf_content = r.get(key).decode('utf-8')
        pdf_library += f"{pdf_name}:\n{pdf_content}\n"
    return pdf_library

async def print_files_in_redis_memory():
    print('\nFiles in Redis memory:')
    keys = vector_db.get_keys_by_prefix('pdf:')
    for key in keys:
        pdf_name = key.split('pdf:')[-1]
        print(f"- {pdf_name}")

def save_history_to_redis(history):
    r.set(CONVERSATION_HISTORY_KEY, history)

def get_history_from_redis():
    history = r.get(CONVERSATION_HISTORY_KEY)
    if history is not None:
        return history.decode('utf-8')
    return ''
    
    

