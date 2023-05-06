import json
from RedisDB import vector_db, text_db, code_db, pdf_db
from Utils import Utils, IngestFiles
from EmbeddingTools import EmbeddingTools
from Config import OUTPUT_DIRECTORY, REPO_DIRECTORY, INGEST_DIRECTORY, CONVERSATION_HISTORY_KEY

class Operations:
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.text_db = TextDatabase()
        self.code_db = CodeDatabase()
        self.pdf_db = PDFDatabase()
        self.utils = Utils()
        self.ingest_files = IngestFiles()
        self.embedding_tools = EmbeddingTools()

    def search_code(self, query):
        return self.code_db.search(query)

    def search_pdfs(self, query):
        return self.pdf_db.search_pdfs(query)

    def search_texts(self, query):
        return self.text_db.search(query)

    def search_vectors(self, query_embedding, top_k=5):
        return self.vector_db.search_vectors(query_embedding, top_k)

    def recommend_vectors(self, item_key, top_k=5):
        return self.vector_db.recommend_vectors(item_key, top_k)

    def ingest_git_repo(self, repo_url):
        file_types = ['.cs', '.html', '.js', '.py']
        return self.ingest_files.ingest_git_repo(repo_url, file_types)

    def ingest_pdf_files(self, directory):
        return self.ingest_files.ingest_pdf_files(directory)

    def print_files_in_redis_memory(self):
        return self.ingest_files.print_files_in_redis_memory()

    def get_similar_texts_based_on_query_embedding(self, query_embedding, top_k=5):
        text_embeddings = list(map(lambda x: [x[0], x[1]], list(self.vector_db.get_all_vectors())))
        return self.embedding_tools.get_similar_texts(query_embedding, text_embeddings, top_k)

    def recommend_based_on_query(self, query, top_k=5):
        texts = list(map(lambda x: [x[0], x[1]], list(self.text_db.get_all_texts())))
        return self.embedding_tools.recommend(query, texts, top_k)

    def get_average_embedding_of_texts(self, texts):
        return self.embedding_tools.get_average_embedding(texts)

    def search_based_on_query(self, query, top_k=5):
        texts = list(map(lambda x: [x[0], x[1]], list(self.text_db.get_all_texts())))
        return self.embedding_tools.search(query, texts, top_k)

class ConversationHistory:
    def __init__(self):
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

class TaskContext:
    # Initialize with a VectorDatabase instance and Utils instance
    def __init__(self, task_id, related_files=None, code_snippets=None, tools_and_methods=None):
        self.task_id = task_id
        self.related_files = related_files if related_files is not None else []
        self.code_snippets = code_snippets if code_snippets is not None else []
        self.tools_and_methods = tools_and_methods if tools_and_methods is not None else {}
        self.vector_db = VectorDatabase()
        self.text_db = TextDatabase()
        self.code_db = CodeDatabase()
        self.pdf_db = PDFDatabase()
        self.utils = Utils()
        self.ingest_files = IngestFiles()
        self.embedding_tools = EmbeddingTools()

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_code_snippet(self, snippet):
        self.code_snippets.append(snippet)

    def add_tool_and_method(self, tool, method):
        if tool not in self.tools_and_methods:
            self.tools_and_methods[tool] = []
        self.tools_and_methods[tool].append(method)

class PlanningContext(TaskContext):  # Inheriting from TaskContext to avoid repetition
    def __init__(self, goal_context, *args, **kwargs):  # Using *args and **kwargs for flexibility
        super().__init__(*args, **kwargs)  # Call parent constructor
        self.goal_context = goal_context
        self.related_files = related_files if related_files is not None else []
        self.relevant_knowledge = relevant_knowledge if relevant_knowledge is not None else {}
        self.tasks = tasks if tasks is not None else []

    def add_related_file(self, file):
        self.related_files.append(file)

    def add_relevant_knowledge(self, knowledge):
        self.relevant_knowledge[knowledge] = True

    def add_task(self, task):
        task.set_plan(self)
        self.tasks.append(task)
        
class Task:
    def __init__(self,
                 goal,
                 task_context=None,
                 specific_instructions=None,
                 methods=None,
                 dependencies=None,
                 context=None):
        self.goal = goal
        self.task_context = task_context if task_context is not None else TaskContext('')
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

class Planning:
    def __init__(self,
                 tasks=None,
                 iterations=1,
                 context=None,
                 planning_context=None):
        self.tasks = tasks if tasks is not None else []
        self.iterations = iterations
        self.context = context if context is not None else {}
        self.planning_context = planning_context if planning_context is not None else PlanningContext('')

    def set_planning_context(self, planning_context):
        planning_context

    def get_planning_context(self):
        return self.planning_context

    def add_task(self, task):
        task.set_plan(self)
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks

    def refine_task(self, task, goal=None, related_files=None, code_snippets=None, specific_instructions=None):
        if goal:
            task.refine_goal(goal)
        if related_files:
            task.update_related_files(related_files)
        if code_snippets:
            task.update_code_snippets(code_snippets)
        if specific_instructions:
            task.update_specific_instructions(specific_instructions)

    def set_iterations(self, iterations):
        self.iterations = iterations

    def update_context(self, context):
        self.context = context

    def execute_finalized_plan(self):
        pass

    def get_sorted_tasks(self):
        return sorted(self.tasks, key=lambda t: (-t.priority, t.goal))

    def get_task_dependencies(self, task):
        return [t for t in self.tasks if t.goal in task.dependencies]

    def set_context(self, context):
        self.context = context

    def add_snippet(self, snippet):
        self.snippets.append(snippet)

    def add_external_link(self, link):
        self.external_links.append(link)

    def set_repo(self, repo):
        self.repo = repo

    def add_highlighted_file(self, file):
        self.highlighted_files.append(file)

    def execute_tasks(self):
        for _ in range(self.iterations):
            sorted_tasks = self.get_sorted_tasks()
            for task in sorted_tasks:
                dependencies = self.get_task_dependencies(task)
                for dependency in dependencies:
                    print(f"Executing dependency: {dependency.goal}")
                    dependency.method()
                print(f"Executing task: {task.goal}")
                task.method()

    def update_planning_context(self, key, value):
        self.planning_context[key] = value
        
