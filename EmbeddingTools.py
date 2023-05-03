import openai
import asyncio
import numpy as np
from sliding_window import SlidingWindowEncoder

class EmbeddingTools:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def get_embeddings(self, texts, model="text-davinci-002"):
        assert isinstance(texts, list), "Input 'texts' should be a list of strings."
        response = openai.Embed.create(model=model, texts=texts)
        return np.array(response["data"])

    def cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_similar_texts(self, query_embedding, text_embeddings, top_k=5):
        similarities = [self.cosine_similarity(query_embedding, text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(index, similarities[index]) for index in sorted_indices[:top_k]]

    def recommend(self, query, texts, top_k=5):
        query_embedding = self.get_embeddings([query])[0]
        text_embeddings = self.get_embeddings(texts)
        return self.get_similar_texts(query_embedding, text_embeddings, top_k)

    def get_average_embedding(self, texts, model="text-davinci-002"):
        embeddings = self.get_embeddings(texts, model)
        return np.mean(embeddings, axis=0)

    def get_nearest_neighbors(self, query_embedding, text_embeddings, top_k=5):
        distances = [np.linalg.norm(query_embedding - text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(distances)
        return [(index, distances[index]) for index in sorted_indices[:top_k]]

    def search(self, query, texts, top_k=5):
        query_embedding = self.get_embeddings([query])[0]
        text_embeddings = self.get_embeddings(texts)
        return self.get_nearest_neighbors(query_embedding, text_embeddings, top_k)

    async def get_ada_embeddings(self, text, model='text-embedding-ada-002'):
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
