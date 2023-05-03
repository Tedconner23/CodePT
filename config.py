import os
import openai
import asyncio

openai.api_key=os.environ['OPENAI_API_KEY']
OUTPUT_DIRECTORY='C:\\Projects\\GPT Output'
ASSISTANT_ROLE='assistant'
SYSTEM_ROLE='system'
ASSISTANT_TRAITS=['helpful','informative','knowledgeable']

def calculate_max_tokens(model: str) -> int:
    model_token_limit = {
        "gpt-3.5-turbo": 4096,
        "text-embedding-ada-002": 4096,
        # Add other models with their token limits here
    }
    used_tokens = 5000
    user_quota = 20000
    remaining_tokens = user_quota - used_tokens
    max_tokens = int(remaining_tokens * 0.8)
    RESERVED_TOKENS = 50
    max_tokens -= RESERVED_TOKENS
    HARD_LIMIT = min(model_token_limit[model], 1000)
    max_tokens = min(max_tokens, HARD_LIMIT)
    return max_tokens

def ensure_output_directory_exists():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

def write_response_to_file(response, filename):
    with open(os.path.join(OUTPUT_DIRECTORY, filename), 'w', encoding='utf-8') as file:
        file.write(response)
		
