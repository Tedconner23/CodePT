import asyncio
import openai
from config import *
from EmbeddingTools import EmbeddingsTools
from Utils import SlidingWindowEncoder, VectorDB
import numpy as np

class GPTInteraction:
    embeddings_tools = EmbeddingsTools(api_key)
    vector_db = VectorDB()
    sliding_window_encoder = SlidingWindowEncoder()

    async def get_optimized_instructions(instruction: str, model: str = 'gpt-3.5-turbo', max_tokens: int = 100) -> str:
        prompt = f"Create an AI understanding and compact instructions for ADAtoperform the following task optimally: {instruction}"
        optimized_instruction = await gpt_interaction(prompt        return optimized_instruction.strip()

    async def gpt_interaction(prompt, model, max_tokens):
        loop = asyncio.get_event_loop()
        RESERVED_TOKENS = 50
        CHUNK_SIZE = max_tokens - RESERVED_TOKENS
        system_message = {'role': SYSTEM_ROLE, 'content': f"You are an {','.join(ASSISTANT_TRAITS)} assistant. Use Ada agents and other tools for GPT-3.5-turbo to aid in interactions and responses."}
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]
        response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(model=model, messages=messages, max_tokens=max_tokens, n=1, stop=None, temperature=.7))
        response_text = response['choices'][0]['message']['content']
        if 'analyze with ada' in response_text.lower():
            analysis_result = await embeddings_tools.get_ada_embeddings(prompt, model)
            response_text += f"\nAda analysis result: {analysis_result}"
        return response_text.strip()

    async def search_content(query: str, top_k: int = 5) -> str:
        query_embedding = await embeddings_tools.get_ada_embeddings(query)
        results = vector_db.search(query_embedding, top_k)
        response = f"Top {top_k} results:\n"
        for i, (key, similarity) in enumerate(results):
            response += f"{i + 1}. {key}: {similarity:.2f}\n"
        return response.strip()

    async def recommend_content(item_key: str, top_k: int = 5) -> str:
        results = vector_db.recommend(item_key, top_k)
        response = f"Top {top_k} recommendations for {item_key}:\n"
        for i, (key, similarity) in enumerate(results):
            response += f"{i + 1}. {key}: {similarity:.2f}\n"
        return response.strip()

    async def get_ada_embeddings(texts: list, model: str = 'text-ada-002') -> np.ndarray:
        encoded_texts = [sliding_window_encoder.encode(text) for text in texts]
        response = await loop.run_in_executor(None, lambda: openai.Embed.create(model=model, texts=encoded_texts))
        return np.array(response['data'])

    async def get_similar_texts(query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        similarities = [embeddings_tools.cosine_similarity(query_embedding, text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(index, similarities[index]) for index in sorted_indices[:top_k]]

    async def recommend(query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await get_ada_embeddings([query])[0]
        text_embeddings = await get_ada_embeddings(texts)
        return await get_similar_texts(query_embedding, text_embeddings, top_k)

    async def get_average_embedding(texts: list, model: str = 'text-ada-002') -> np.ndarray:
        embeddings = await get_ada_embeddings(texts, model)
        return np.mean(embeddings, axis=0)

    async def get_nearest_neighbors(query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        distances = [np.linalg.norm(query_embedding - text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(distances)
        return [(index, distances[index]) for index in sorted_indices[:top_k]]

    async def search(query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await get_ada_embeddings([query])[0]
        text_embeddings = await get_ada_embeddings(texts)
        return await get_nearest_neighbors(query_embedding, text_embeddings, top_k)
