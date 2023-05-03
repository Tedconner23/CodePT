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
    def __init__(self, name, description, keywords, method):
        self.name = name
        self.description = description
        self.keywords = keywords
        self.method = method

class Planning:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def get_tasks(self):
        return self.tasks
