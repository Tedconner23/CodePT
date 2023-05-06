import asyncio
import os
import re
from GPT import GPTInteraction
from RedisDB import *
from Config import *
from DataStructures import *
from sklearn.metrics.pairwise import cosine_similarity
from Interactor import process_redis_memory_context, get_memory_keywords, keywordizer
from PlanTask import TaskContext, PlanningContext, Task, Planning

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
            if sub_choice == 'a':
                repo_url = input('Enter the Git repository URL:')
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