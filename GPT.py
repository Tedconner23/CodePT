import asyncio
import openai
from Config import *
from EmbeddingTools import EmbeddingTools
from Utils import *
from RedisDB import vector_db, text_db, code_db, pdf_db
from ChatHistory import ConversationHistory, CONVERSATION_HISTORY_KEY
from PlanTask import TaskContext, PlanningContext, Task, Planning
import numpy as np

class GPTInteraction:
    def __init__(self, model_data):
        self.model_data = model_data
        self.embedding_tools = EmbeddingTools(api_key)
        self.utils = Utils()
        self.vector_db = VectorDB()
    
    async def get_embeddings(self, text, model_name):
        model = self.model_data[model_name]
        loop = asyncio.get_event_loop()
        window_size = model['window_size']
        step_size = model['step_size']
        sliding_window_encoder = SlidingWindowEncoder(window_size, step_size)
        windows = sliding_window_encoder.encode(text)

        async def get_embeddings(window):
            response = await loop.run_in_executor(None, lambda: openai.Completion.create(engine=model_name, prompt=window, max_tokens=1, n=1, stop=None, temperature=.5))
            return response.choices[0].text.strip()

        embeddings = await asyncio.gather(*(get_embeddings(window) for window in windows))
        return embeddings

    async def user_to_gpt_interaction(self, prompt, model_name):
        model = self.model_data[model_name]
        max_tokens = self.calculate_max_tokens(model)
        RESERVED_TOKENS = 50
        CHUNK_SIZE = max_tokens - RESERVED_TOKENS

        system_message = {'role': 'system', 'content': f"You are an {','.join(ASSISTANT_TRAITS)} assistant. Use Ada agents and other tools for GPT-3.5-turbo to aid in interactions and responses."}
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]

        response = await self._execute_chat_completion(model_name, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']

        return response_text.strip()

    async def gpt_to_gpt_chat_interaction(self, prompt, model, max_tokens, toAgent, toAgentHistory, fromAgent, fromAgentHistory):
        messages = [
            {'role': 'user', 'content': f"You are asking a question of, or commanding, {toAgent} to resolve: {prompt}"},
            {'role': 'system', 'content': f"You are interacting with {toAgent}. The prompt to resolve is: {prompt}", 'traits': '', 'name': 'fromAgent', 'history': 'fromAgentHistory', 'interlocutor': 'toAgent', 'tools_and_methods': 'Available tools to call: tools_and_methods(reduced by context)'},
            {'role': 'assistant', 'content': f"You are interacting with {toAgent}. The prompt to resolve is: {prompt}", 'traits': '', 'name': 'toAgent', 'history': 'toAgentHistory', 'interlocutor': 'fromAgentHistory', 'tools_and_methods': 'Available tools to call: tools_and_methods(reduced by context)'}
        ]

        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def complete_prompt(self, prompt, model, max_tokens, message):
        messages = [{'role': 'user', 'content': f"As a basic prompt completion bot, you are with a variation of the message: {message}. Keep meaning but the language does not have to be exact."}]
        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def respond_to_prompt(self, prompt, model, max_tokens, history, string_in, task_in, personality, name, responsibility):
        m1 = [{'role': 'user', 'content': f"Generate prompt instructions for text-davinci-003 where the instructions are: You are {name} and have personality: {personality} with task: {task_in} as context."}]
        response1 = await self._execute_chat_completion(model, m1, max_tokens)
        generated_prompt = response1['choices'][0]['message']['content']

        m2 = [{'role': 'user', 'content': f"You are replying to the user after post analysis. Your response prompt is: {generated_prompt}"}]
        response2 = await self._execute_chat_completion(model, m2, max_tokens)
        response_text = response2['choices'][0]['message']['content']
        return response_text.strip()

    async def gpt_to_user_interaction(self, prompt, model, max_tokens):
        RESERVED_TOKENS = 50
        CHUNK_SIZE = max_tokens - RESERVED_TOKENS

        system_message = {'role': 'system', 'content': f"You are an {','.join(ASSISTANT_TRAITS)} assistant. Use Ada agents and other tools for GPT-3.5-turbo to aid in interactions and responses."}
        user_message = {'role': 'user', 'content': prompt}
        messages = [system_message, user_message]

        response = await self._execute_chat_completion(model, messages, max_tokens)
        response_text = response['choices'][0]['message']['content']
        return response_text.strip()

    async def _execute_chat_completion(self, model_name, messages, max_tokens):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: openai.ChatCompletion.create(model=model_name, messages=messages, max_tokens=max_tokens, n=1, stop=None, temperature=.7))
        return response

    @staticmethod
    def calculate_max_tokens(model: dict, remaining_tokens: int) -> int:
        model_token_limit = int(remaining_tokens * .8)
        RESERVED_TOKENS = 50
        max_tokens = model_token_limit - RESERVED_TOKENS
        HARD_LIMIT = min(model['token_limits'], 1000)
        max_tokens = min(max_tokens, HARD_LIMIT)
        return max_tokens

class SlidingWindowEncoder:
    def __init__(self, window_size, step_size):
        self.window_size = window_size
        self.step_size = step_size

    def encode(self, input_data):
        if len(input_data) <= self.window_size:
            return [input_data]
        windows = []
        start = 0
        end = self.window_size
        while end <= len(input_data):
            windows.append(input_data[start:end])
            start += self.step_size
            end += self.step_size
        if start < len(input_data):
            windows.append(input_data[start:])
        return windows

    def decode(self, encoded_data):
        decoded_data = []
        for window in encoded_data:
            decoded_data.extend(window)
        return decoded_data    
