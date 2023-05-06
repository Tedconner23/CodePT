import openai
import asyncio
import numpy as np
from GPT import SlidingWindowEncoder

class EmbeddingTools:
    def __init__(self, gpt_interaction):
        self.gpt_interaction = gpt_interaction

    async def cosine_similarity(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the cosine similarity between two vectors {a} and {b}.")

    async def euclidean_distance(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Euclidean distance between two vectors {a} and {b}.")

    async def manhattan_distance(self, a, b):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Manhattan distance between two vectors {a} and {b}.")

    async def normalize_embedding(self, embedding):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Normalize the given vector {embedding}.")

    async def get_similar_texts(self, query_embedding, text_embeddings, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} similar texts for the given query embedding {query_embedding} and text embeddings {text_embeddings}.")

    async def get_similar_texts_custom_metric(self, query_embedding, text_embeddings, metric_function, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} similar texts for the given query embedding {query_embedding} and text embeddings {text_embeddings} using the custom metric function {metric_function}.")

    async def recommend(self, query, texts, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Recommend the top {top_k} texts for the given query {query} and texts {texts}.")

    async def get_average_embedding(self, texts, model='text-davinci-002'):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the average embedding for the given texts {texts} using the model {model}.")

    async def get_nearest_neighbors(self, query_embedding, text_embeddings, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} nearest neighbors for the given query embedding {query_embedding} and text embeddings {text_embeddings}.")

    async def search(self, query, texts, top_k=5):
        return await self.gpt_interaction.user_to_gpt_interaction(f"Search the top {top_k} texts for the given query {query} and texts {texts}.")

    async def search_content(self, query: str, top_k: int = 5) -> str:
        return await self.gpt_interaction.user_to_gpt_interaction(f"Search the top {top_k} content for the given query {query}.")

    async def recommend_content(self, item_key: str, top_k: int = 5) -> str:
        return await self.gpt_interaction.user_to_gpt_interaction(f"Recommend the top {top_k} content for the given item key {item_key}.")
        
    async def get_similar_texts(self, gpt_interaction, query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        similarities = [gpt_interaction.embeddings_tools.cosine_similarity(query_embedding, text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(similarities)[::-1]
        return [(index, similarities[index]) for index in sorted_indices[:top_k]]

    async def recommend_based_on_query(self, gpt_interaction, query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await gpt_interaction.embeddings_tools.get_ada_embeddings([query])[0]
        text_embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts)
        return await self.get_similar_texts(gpt_interaction, query_embedding, text_embeddings, top_k)

    async def get_average_embedding(self, gpt_interaction, texts: list, model: str = 'text-ada-002') -> np.ndarray:
        embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts, model)
        return np.mean(embeddings, axis=0)

    async def get_nearest_neighbors(self, gpt_interaction, query_embedding: np.ndarray, text_embeddings: list, top_k: int = 5) -> list:
        distances = [np.linalg.norm(query_embedding - text_embedding) for text_embedding in text_embeddings]
        sorted_indices = np.argsort(distances)
        return [(index, distances[index]) for index in sorted_indices[:top_k]]

    async def search_based_on_query(self, gpt_interaction, query: str, texts: list, top_k: int = 5) -> list:
        query_embedding = await gpt_interaction.embeddings_tools.get_ada_embeddings([query])[0]
        text_embeddings = await gpt_interaction.embeddings_tools.get_ada_embeddings(texts)
        return await self.get_nearest_neighbors(gpt_interaction, query_embedding, text_embeddings, top_k)
        
    def unique_values(self, column_name):
        data = self.read_data()
        return data[column_name].unique()

    def basic_statistics(self, column_name):
        data = self.read_data()
        return data[column_name].describe()

    def top_n_most_frequent(self, column_name, n=10):
        data = self.read_data()
        return data[column_name].value_counts().nlargest        
        
    async def process_redis_memory_context():
        keys = vector_db.get_keys_by_prefix('repo:') + vector_db.get_keys_by_prefix('pdf:')
        embeddings = [np.array(vector_db.get_item(key), dtype=float) for key in keys]
        ada_embeddings = await get_ada_embeddings([key for key in keys])
        return dict(zip(keys, ada_embeddings))

    async def get_memory_keywords(redis_memory_context, ada_embeddings, threshold=0.8):
        memory_keywords = []
        for key, value in redis_memory_context.items():
            similarity = cosine_similarity(ada_embeddings, value)
            if similarity >= threshold:
                memory_keywords.append(key)
        return memory_keywords

    async def keywordizer(planning, input_text, redis_memory_context):
        tasks_to_execute = []
        ada_embeddings = await get_ada_embeddings(input_text)
        memory_keywords = await get_memory_keywords(redis_embeddings)

        for task in planning.get_tasks():
            all_keywords = task.keywords + memory_keywords
            if any(re.search(f"\\b{keyword}\\b", input_text, re.IGNORECASE) for keyword in all_keywords):
                tasks_to_execute.append(task)
            else:
                inferred_keywords = await get_similar_keywords(ada_embeddings, task.keywords)
                if inferred_keywords:
                    tasks_to_execute.append(task)

        return tasks_to_execute