import asyncio
from gpt import gpt_interaction, SlidingWindowEncoder
from redismem import (
    clear_redis_memory,
    ingest_git_repo,
    ingest_pdf_files,
    print_files_in_redis_memory,
    get_history_from_redis,
    save_history_to_redis,
)
from config import write_response_to_file, calculate_max_tokens
from datastructures import Planning, Task
import re

def keywordizer(planning, input_text):
    tasks_to_execute = []
    for task in planning.get_tasks():
        if any(
            re.search(f'\\b{keyword}\\b', input_text, re.IGNORECASE)
            for keyword in task.keywords
        ):
            tasks_to_execute.append(task)
    return tasks_to_execute

async def main():
    planning = Planning()

    while True:
        print('\nOptions:')
        print('1. Clear Redis memory')
        print('2. Ingest files into memory')
        print('3. Print files in Redis memory')
        print('4. Continue with GPT interaction')
        print('5. Exit')

        choice = input('Enter your choice (1/2/3/4/5): ')

        if choice == '1':
            clear_redis_memory()
            print('Redis memory cleared.')

        elif choice == '2':
            print('Options:')
            print('  a. Ingest Git repository')
            print('  b. Ingest PDF files')
            sub_choice = input('Enter your choice (a/b): ')

            if sub_choice.lower() == 'a':
                repo_url = input('Enter the Git repository URL: ')
                await ingest_git_repo(repo_url)
            elif sub_choice.lower() == 'b':
                directory = input('Enter the directory containing PDF files: ')
                await ingest_pdf_files(directory)
            else:
                print('Invalid choice. Please try again.')

        elif choice == '3':
            await print_files_in_redis_memory()

        elif choice == '4':
            conversation_history = get_history_from_redis()

            while True:
                new_input = input('User: ')
                if new_input.lower() == 'exit':
                    break

                tasks = keywordizer(planning, new_input)
                for task in tasks:
                    await planning.execute_task(task)

                combined_input = (
                    f"{conversation_history}\nUser: {new_input}\nAI: "
                )
                max_tokens = calculate_max_tokens('gpt-3.5-turbo')
                response = await gpt_interaction(
                    combined_input, 'gpt-3.5-turbo', max_tokens
                )

                conversation_history += f"User: {new_input}\nAI: {response}"
                save_history_to_redis(conversation_history)
                print(f"AI: {response}")

                if 'output to file' in new_input.lower():
                    file_name = input('Enter the output file name: ')
                    write_response_to_file(response, file_name)

        elif choice == '5':
            break

        else:
            print('Invalid choice. Please again.')

if __name__ == '__main__':
    asyncio.run(main())
