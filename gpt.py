import asyncio
import os
import re
from gpt import gpt_interaction, SlidingWindowEncoder
from redismem import clear_redis_memory, ingest_git_repo, ingest_pdf_files, print_files_in_redis_memory, get_history_from_redis, save_history_to_redis
from config import write_response_to_file, calculate_max_tokens
from datastructures import Planning, Task
from sklearn.metrics.pairwise import cosine_similarity


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


async def main():
    planning = Planning()
    redis_memory_context = await process_redis_memory_context()

    while True:
        display_main_menu()
        choice = input('Enter your choice (1/2/3/4/5/6/7):')
        
        if choice == '1':
            clear_redis_memory()
            print('Redis memory cleared.')
        elif choice == '2':
            display_sub_menu()
            sub_choice = input('Enter your choice (a/b):')
            = input('Enter the directory containing PDF files:')
                await ingest_pdf_files(directory)
                print('PDF files ingested successfully.')
            else:
                print('Invalid choice. Please try again.')
        elif choice == '3':
            await print_files_in_redis_memory()
        elif choice == '4':
            conversation_history = get_history_from_redis()
            print("Enter 'exit' to stop the GPT interaction.")
            while True:
                new_input = input('User:')
                if new_input.lower() == 'exit':
                    break
                tasks = await keywordizer(planning, new_input, redis_memory_context)
                for task in tasks:
                    await planning.execute_task(task)
                combined_input = f"{conversation_history}\nUser:{new_input}\nAI:"
                max_tokens = calculate_max_tokens('gpt-3.5-turbo')
                response = await gpt_interaction(combined_input, 'gpt-3.5-turbo', max_tokens)
                conversation_history += f"User:{new_input}\nAI:{response}"
                save_history_to_redis(conversation_history)
                print(f"AI:{response}")
        elif choice == '5':
            print('Please provide the script name (script_name.py arg1 arg2 ...)')
            list_scripts()
            script_input = input('Enter script name and arguments:')
            try:
                script_args = script_input.split()
                script_name = script_args.pop(0)
                if script_name.endswith('.py') and os.path.exists(script_name):
                    await asyncio.create_subprocess_exec('python', script_name, *script_args)
                    print(f"Successfully executed {script_name} with arguments: {', '.join(script_args)}")
                else:
                    print(f"Script '{script_name}' not found. Please choose a valid script.")
            except Exception as e:
                print(f"Error executing the script: {e}")
        elif choice == '6':
            try:
                await asyncio.create_subprocess_exec('python', 'main.py')
                print('Successfully executed main.py script.')
            except Exception as e:
                print(f"Error executing the main.py script: {e}")
        elif choice == '7':
            print('Exiting...')
            break
        else:
            print('Invalid choice. Please try again.')


if __name__ == '__main__':
    asyncio.run(main())
