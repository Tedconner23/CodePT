import json
import os
import git
import shutil
from RedisDB import vector_db, text_db, code_db, pdf_db
from EmbeddingTools import EmbeddingTools
from Config import 
                    
class Operations:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.text_db = TextDatabase()
        self.code_db = CodeDatabase()
        self.pdf_db = PDFDatabase()
        self.utils = Utils(self)
        self.ingest_files = IngestFiles(self)
        self.embedding_tools = EmbeddingTools(self)
        self.task_manager = TaskManager()

    def search(self, query, search_type='code'):
        if search_type == 'code':
            return self.code_db.search(query)
        elif search_type == 'pdfs':
            return self.pdf_db.search_pdfs(query)
        elif search_type == 'texts':
            return self.text_db.search(query)

    def recommend(self, item_key, top_k=5, recommend_type='vectors'):
        if recommend_type == 'vectors':
            return self.vector_db.recommend_vectors(item_key, top_k)  

    def get_average_embedding_of_texts(self, texts):
        return self.embedding_tools.get_average_embedding(texts)

    async def recommend_based_on_query(self, query, top_k=5):
        texts = list(map(lambda x: [x[0], x[1]], list(self.text_db.get_all_texts())))
        return await self.embedding_tools.recommend(query, texts, top_k)

    async def search_based_on_query(self, query, top_k=5):
        texts = list(map(lambda x: [x[0], x[1]], list(self.text_db.get_all_texts())))
        return await self.embedding_tools.search(query, texts, top_k)
        
    def add_task(self, task):
        self.task_manager.add_task(task)

    def get_tasks(self):
        return self.task_manager.get_tasks()

    def refine_task(self, task_id, goal=None, related_files=None, code_snippets=None, specific_instructions=None):
        self.task_manager.refine_task(task_id, goal, related_files, code_snippets, specific_instructions)

    def execute_tasks(self, iterations=1):
        self.task_manager.execute_tasks(iterations)
        
    class IngestFiles:
        def __init__(self, parent_operations):
            self.parent_operations = parent_operations

        def ingest_git_repo(self, repo_url, redis_db, file_types=None):
            if file_types is None:
                file_types = ['.cs', '.html', '.js', '.py']

            temp_dir = "temp_git_repo"
            git.Repo.clone_from(repo_url, temp_dir)

            for root, _, files in os.walk(temp_dir):
                for file in files:
                    if any(file.endswith(ext) for ext in file_types):
                        file_path = os.path.join(root, file)
                        with open ('r') as f:
                            content = f.read()
                            key = f"{os.path.basename(file)}_{os.path.splitext(file)[0]}"
                            redis_db.set(key, content)

            shutil.rmtree(temp_dir)

        def ingest_folder(self, folder_path, redis_db, specific_file=None):
            if specific_file:
                file_path = os.path.join(folder_path, specific_file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    key = f"{os.path.basename(specific_file)}_{os.path.splitext(specific_file)[0]}"
                    redis_db.set(key, content)
            else:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            content = f.read()
                            key = f"{os.path.basename(file)}_{os.path.splitext(file)[0]}"
                            redis_db.set(key, content)
                        
    class Utils:
        def __init__(self, parent_operations):
            self.parent_operations = parent_operations

        def download_data(self, data_url, data_path, file_name):
            zip_file_path = os.path.join(data_path, file_name + '.zip')
            csv_file_path = os.path.join(data_path, file_name + '.csv')
            if os.path.isfile(csv_file_path):
                print('File already downloaded')
            elif os.path.isfile(zip_file_path):
                print('Zip downloaded but not unzipped, unzipping now...')
                self.extract_zip(zip_file_path)
            else:
                print('File not found, downloading now...')
                wget.download(data_url, out=data_path, bar=True)
                self.extract_zip(zip_file_path)

        def extract_zip(self, zip_file_path):
            with zipfile.ZipFile(zip_file_path,'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file_path))
                os.remove(zip_file_PATH)
                print(f"File downloaded to {os.path.dirname(zip_file_PATH)}")

        def load_data_to_databases(self, data_rows):
            for row in data_rows:
                item_id = str(row.get('id'))
                if item_id:
                    title = row.get('title')
                    code = row.get('code')
                    pdf = row.get('pdf')
                    content_vector = row.get('content_vector')

                    if title:
                        text_db.add_item(item_id, title)
                    if code:
                        code_db.add_item(item_id, code)
                    if pdf:
                        pdf_db.add_item(item_id, pdf)
                    if content_vector:
                        vector_db.add_item(item_id,content_vector)

        def get_item_by_id(self, item_id, database):
            return database.get_item(item_id)

        def search_item(self, database, payload, top_k=5):
            return database.search(payload, top_k=top_k)

        def recommend_similar_items(self, database, item_id, top_k=5):
            return database.recommend_vectors(item_id, top_k=top_k)

        def add_chat_history(self, input_data, output_data, database):
            database.add_chat(input_data,output_data)

        def get_full_chat_history(self, database):
            return database.get_full_history()

        def add_summarized_history(self, summarized_input,summarized_output,database):
            database.add_summary(summarized_input,summarized_output)

        def get_summarized_chat_history(self,database):
            return database.get_summarized_history()

        async def summarize_chat_history(self,gpt_interaction_model:object,historical_chats:list,model)->list:
            summarized_chats=[]
            for (chat_input,response) in historical_chats:
                summarized_input=await gpt_interaction_model.summarize_text(chat_input,model)
                summarized_response=await gpt_interaction_model.summarize_text(response,model)
                summarized_chats.append((summarized_input,summarized_response))
            return summarized_chats

    class EmbeddingTools:
        def __init__(self, parent_operations):
            self.parent_operations = parent_operations
            self.gpt_interaction = GPTInteraction()
            
        def unique_values(self, column_name):
            data = self.read_data()
            return data[column_name].unique()

        def basic_statistics(self, column_name):
            data = self.read_data()
            return data[column_name].describe()

        def top_n_most_frequent(self, column_name, n=10):
            data = self.read_data()
            return data[column_name].value_counts().nlargest 

        @staticmethod
        async def cosine_similarity(self, a, b):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the cosine similarity between two vectors {a} and {b}.")

        @staticmethod
        async def euclidean_distance(self, a, b):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Euclidean distance between two vectors {a} and {b}.")

        @staticmethod
        async def manhattan_distance(self, a, b):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Calculate the Manhattan distance between two vectors {a} and {b}.")

        @staticmethod
        async def normalize_embedding(embedding):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Normalize the given vector {embedding}.")    
     

        async def get_average_embedding(self, texts, model='text-davinci-002'):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Get the average embedding for the given texts {texts} using the model {model}.")

        async def get_nearest_neighbors(self, query_embedding, text_embeddings, top_k=5):
            return await self.gpt_interaction.user_to_gpt_interaction(f"Get the top {top_k} nearest neighbors for the given query embedding {query_embedding} and text embeddings {text_embeddings}.")

        async def recommend(self, query, texts, top_k=5):
            query_embedding = await self.gpt_interaction.get_text_embedding(query)
            text_embeddings = [(text_id, await self.gpt_interaction.get_text_embedding(text)) for text_id, text in texts]
            
            return await get_similar_texts(query_embedding, text_embeddings, top_k)

        async def search(self, query, texts, top_k=5):
            query_embedding = await self.gpt_interaction.get_text_embedding(query)
            text_embeddings = [(text_id, await self.gpt_interaction.get_text_embedding(text)) for text_id, text in texts]
            
            return await get_similar_texts(query_embedding, text_embeddings, top_k)

        async def get_similar_texts(self, query_embedding, text_embeddings, top_k=5):
            similarities = [await cosine_similarity(query_embedding, text_embedding) for _, text_embedding in text_embeddings]
            sorted_indices = np.argsort(similarities)[::-1]
            
            return [(text_id, similarities[index]) for index, (text_id, _) in enumerate(text_embeddings) if index in sorted_indices[:top_k]]

class ConversationHistory:
    def __init__(self, operations):
        self.operations = operations
        self.interactions = []
        self.documents = []
        self.queries = []

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def add_document(self, document):
        self.documents.append(document)

    def add_query(self, query):
        self.queries.append(query)

    def get_interactions(self):
        return self.interactions

    def get_documents(self):
        return self.documents

    def get_queries(self):
        return self.queries

    def save_to_redis(self, r):
        r.set(CONVERSATION_HISTORY_KEY, self.serialize())

    def load_from_redis(self, r):
        history_data = r.get(CONVERSATION_HISTORY_KEY)
        if history_data is not None:
            self.deserialize(history_data.decode('utf-8'))

    def serialize(self):
        return json.dumps({'interactions': self.interactions,
                           'documents': self.documents,
                           'queries': self.queries})

    def deserialize(self, data):
        history_data = json.loads(data)
        self.interactions = history_data['interactions']
        self.documents = history_data['documents']
        self.queries = history_data['queries']

    def clear_history(self, r):
        self.interactions = []
        self.documents = []
        self.queries = []
        r.delete(CONVERSATION_HISTORY_KEY)
   
class TaskManager:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        task.set_task_manager(self)
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks

    def refine_task(self, task_id, goal=None, related_files=None, code_snippets=None, specific_instructions=None):
        task = self._find_task_by_id(task_id)
        if task:
            task.update_goal(goal)
            task.update_related_files(related_files)
            task.update_code_snippets(code_snippets)
            task.update_specific_instructions(specific_instructions)

    def _find_task_by_id(self, task_id):
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        return None
        
    def remove_task(self, task_id):
        task = self._find_task_by_id(task_id)
        if task:
            self.tasks.remove(task)

    def execute_tasks(self, iterations=1):
        for _ in range(iterations):
            for task in self.tasks:
                print(f"Executing task: {task.goal}")
                task.execute()
   
class Task:
    def __init__(self, goal, task_id=None, related_files=None, code_snippets=None, tools_and_methods=None, specific_instructions=None, methods=None, dependencies=None, context=None):
        self.goal = goal
        self.task_id = task_id
        self.related_files = related_files if related_files is not None else []
        self.code_snippets = code_snippets if code_snippets is not None else []
        self.tools_and_methods = tools_and_methods if tools_and_methods is not None else {}
        self.specific_instructions = specific_instructions if specific_instructions is not None else []
        self.methods = methods if methods is not None else []
        self.dependencies = dependencies if dependencies is not None else []
        self.context = context if context is not None else {}
        
    def set_task_context(self, task_context):
        self.task_context = task_context

    def get_task_context(self):
        return self.task_context

    def update_goal(self, new_goal):
        self.goal = new_goal
        
    def update_related_files(self, files):
        if files:
            self.related_files = files

    def update_code_snippets(self, snippets):
        if snippets:
            self.code_snippets = snippets

    def update_specific_instructions(self, instructions):
        if instructions:
            self.specific_instructions = instructions
            
    def set_task_manager(self, task_manager):
        self.task_manager = task_manager

    def update_goal(self, new_goal):
        if new_goal:
            self.goal = new_goal

    def __repr__(self):
        return f"Task(id={self.task_id}, goal='{self.goal}')"

    def execute(self):
        # Implement the task execution logic here
        for method in self.methods:
            print(f"Executing method: {method.__name__}")
            try:
                method(self)
            except Exception as e:
                print(f"Error executing method {method.__name__}: {e}")

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_code_snippet(self, snippet):
        self.code_snippets.append(snippet)

    def add_specific_instruction(self, instruction):
        self.specific_instructions.append(instruction)

    def add_method(self, method):
        self.methods.append(method)

    def refine_goal(self, new_goal):
        self.methods = new_goal

    def update_related_files(self, files):
        self.related_files = files

    def update_code_snippets(self, snippets):
        self.code_snippets = snippets

    def update_specific_instructions(self, instructions):
        self.specific_instructions = instructions

    def present_and_review_methods(self, methods):
        self.methods = methods

    def update_task_context(self, key, value):
        self.task_context[key] = value

