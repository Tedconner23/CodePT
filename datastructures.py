import json
import redis

tools_and_methods = {   'EmbeddingsTools': {   'methods': [   'get_ada_embeddings',   'cosine_similarity',   'get_similar_texts',   'recommend',   'get_average_embedding',   'get_nearest_neighbors',   'search'   ]   },   'GPTInteraction': {   'methods': [   'get_optimized_instructions',   'gpt_interaction',   'search_content',   'recommend_content',   'get_ada_embeddings',   'get_similar_texts',   'recommend',   'get_average_embedding',   'get_nearest_neighbors',   'search'   ]   },   'VectorDatabase': {   'methods': [   'add_item',   'get_item',   'get_keys_by_prefix',   'search',   'recommend'   ]   },   'Ingestion': {   'methods': [   'get_ada_embeddings',   'ingest_git_repo',   'ingest_pdf_files',   'get_pdf_library',   'print_files_in_redis_memory',   'save_history_to_redis',   'get_history_from_redis'   ]   },   'SlidingWindowEncoder': {   'methods': [   'encode',   'decode'   ]   },   'ExternalResources': {   'links': [   'https://platform.openai.com/docs/guides/embeddings/what-are-embeddings'   ]   } }

CONVERSATION_HISTORY_KEY = 'conversation_history'
r = redis.Redis(host='localhost', port=6379, db=0)

class ConversationHistory:
    def __init__(self):
        self.interactions = []
        self.tasks = []
        self.documents = []
        self.queries = []

    def add_interaction(self, interaction):
        self.interactions.append(interaction)

    def add_task(self, task):
        self.tasks.append(task)

    def add_document(self, document):
        self.documents.append(document)

    def add_query(self, query):
        self.queries.append(query)

    def get_interactions(self):
        return self.interactions

    def get_tasks(self):
        return self.tasks

    def get_documents(self):
        return self.documents

    def get_queries(self):
        return self.queries

    def save_to_redis(self):
        r.set(CONVERSATION_HISTORY_KEY, self.serialize())

    def load_from_redis(self):
        history_data = r.get(CONVERSATION_HISTORY_KEY)
        if history_data is not None:
            self.deserialize(history_data.decode('utf-8'))

    def serialize(self):
        return json.dumps({
            'interactions': self.interactions,
            'tasks': self.tasks,
            'documents': self.documents,
            'queries': self.queries
        })

    def deserialize(self, data):
        history_data = json.loads(data)
        self.interactions = history_data['interactions']
        self.tasks = history_data['tasks']
        self.documents = history_data['documents']
        self.queries = history_data['queries']

    def clear_history(self):
        self.interactions = []
        self.tasks = []
        self.documents = []
        self.queries = []
        r.delete(CONVERSATION_HISTORY_KEY)
        
class Task:
    def __init__(self, goal, related_files=None, code_snippets=None, specific_instructions=None, methods=None, dependencies=None):
        self.goal = goal
        self.related_files = related_files if related_files is not None else []
        self.code_snippets = code_snippets if code_snippets is not None else []
        self.specific_instructions = specific_instructions if specific_instructions is not None else []
        self.methods = methods if methods is not None else []
        self.dependencies = dependencies if dependencies is not None else []

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


class Planning:
    def __init__(self, context="", repo="", iterations=1):
        self.tasks = []
        self.context = context
        self.snippets = []
        self.external_links = []
        self.repo = repo
        self.highlighted_files = []
        self.iterations = iterations

    def add_task(self, task):
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks

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
