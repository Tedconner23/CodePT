from Operations import TaskContext, PlanningContext, Task, Planning, Operations, ConversationHistory
from GPT import GPTInteraction
from Config import OUTPUT_DIRECTORY, REPO_DIRECTORY, INGEST_DIRECTORY, CONVERSATION_HISTORY_KEY, agent_models, model_data
from EmbeddingTools import embedding_tools_methods
import multiprocessing

# Executive:
# ManagerAgent: Coordinates AI system and agents. Collaborates with basic agents, mainly RelayAgent. Initializes ProjectManagerAgent, EmbedAgent, TerminalAgent, Browse&FileAgent, and RelayAgent. Registers agents with RelayAgent.
# ProjectManagerAgent: Focuses on user goals/tasks. Initializes with Manager. Spawns LeadAgent for specific purposes or PersonaAgent for non-specific tasks. Also initializes MemoryAgent.
# PersonaAgent: Manages user interactions and worker agents for requests. Acts as an extension of ProjectManagerAgent, handling user requests directly and spawning task-specific agents.

# Leads:
# LeadAgent: Collaborates with ProjectManagerAgent and user to design tasks (e.g., methods, story chapters). Provides contextual recommendations. Initializes GitManAgent and CodeAssemblyAgent for planning sessions.
# MemoryAgent: Uses EmbedAgent to retrieve embeddings, selecting queries intelligently. Directs user input post-intent analysis for context analysis. Retrieves related context from memory or cached/stored links.

# Workers:
# GitManAgent: Manages Git operations, including commits, branches, and merges.
# CodeAssemblyAgent: Processes plans with tasks defining method names and creation prompts. Performs syntax analysis and input-output analysis after method completion. Ensures code quality and adherence to guidelines.
# TextAssemblyAgent: Similar to CodeAssemblyAgent, but works on a chapter or paragraph basis. Focuses on content coherence, flow, and language quality.

# Basic:
# EmbedAgent: Handles embedding files to databases and retrieving comparisons. Utilizes advanced similarity algorithms to compare and match data.
# TerminalAgent: Manages basic input-output communication, displaying messages to users. Batches intent analysis and context searching based on intent. Supports user assistance and troubleshooting.
# RelayAgent: Manages communication and dequeuing. Redirects communication based on lead or executive prompts. Ensures efficient routing and prioritization of tasks.
# Browse&FileAgent: Manages local file system operations, browsing, and web downloads. Provides local links to embed models. Writes browsed files and downloaded resources to local files. Handles file organization, search, and retrieval.


class Agent:
    def __init__(self, role, level, traits, tools, agent_id):
        self.role = role
        self.level = level
        self.traits = traits
        self.tools = tools
        self.agent_id = agent_id
        self.name = f"{role}{agent_id}"
        self.gpt_interaction = GPTInteraction(model_data)
        self.processes = []

    async def engage_user(self):
        prompt = "Hello! I am your AI assistant. How can I help you today?"
        model_name = "text-davinci-002"
        response_text = await self.gpt_interaction.user_to_gpt_interaction(prompt, model_name)
        print(f"User engaged by {self.name}: {response_text}")

    async def notify_admin(self):
        prompt = "An error has occurred. Please inform the administrator."
        model_name = "text-davinci-002"
        response_text = await self.gpt_interaction.user_to_gpt_interaction(prompt, model_name)
        print(f"Admin notified by {self.name}: {response_text}")

    async def execute_task(self):
        if self.role == 'Browsing' or self.role == 'Code':
            prompt = "Please browse for relevant information or write code to complete the task."
            model_name = "text-davinci-002"
            response_text = await self.gpt_interaction.user_to_gpt_interaction(prompt, model_name)
            print(f"Task executed by {self.name}: {response_text}")

    async def halt_execution(self):
        prompt = "The execution has been halted. Please stop working on the current task."
        model_name = "text-davinci-002"
        response_text = await self.gpt_interaction.user_to_gpt_interaction(prompt, model_name)
        print(f"Execution halted by {self.name}: {response_text}")

    async def terminate_agent_self(self):
        prompt = f"Agent {self.name} is being terminated."
        model_name = "text-davinci-002"
        response_text = await self.gpt_interaction.user_to_gpt_interaction(prompt,model_name)
        print(f"Agent {self.name} self-terminated: {response_text}")

    def spawn_worker(self, worker_class,*args):
        if self.level == 'executive' or self.level == 'lead':
            process=multiprocessing.Process(target=worker_class,args=args)
            process.start()
            self.processes.append(process)

    def spawn_worker(self, worker_class, *args):
        if self.level == 'executive' or self.level == 'lead':
            process = multiprocessing.Process(target=worker_class, args=args)
            process.start()
            self.processes.append(process)

    def add_subagent(self, subagent_class, *args):
        if self.level == 'executive':
            subagent = subagent_class(*args)
            self.subagents.append(subagent)
            return subagent

    def subagent__receive(self, message):
        print(f"{self.name} received message: {message}")

    def subagent__send(self, message, target_agent):
        print(f"{self.name} sent message to {target_agent.name}: {message}")
        target_agent.subagent__receive(message)
        
    def exec_decide(self, worker_class, *args):
        if self.level == 'executive':
            for subagent in self.subagents:
                if isinstance(subagent, worker_class):
                    subagent.execute_task(*args)

    def exec_run_method(self, worker_class, method_name, *args):
        if self.level == 'executive':
            for subagent in self.subagents:
                if isinstance(subagent, worker_class) and hasattr(subagent, method_name):
                    method = getattr(subagent, method_name)
                    method(*args)

    def self_assessment(self):
        print(f"Self-assessment performed by {self.name}")

    def exec_run_command(self, command, *args, **kwargs):
        if self.level == 'executive':
            method = getattr(self, command)
            method(*args, **kwargs)

    def terminate_all_agents(self):
        if self.level == 'executive':
            for process in self.processes:
                process.terminate()
            print("All agents terminated")

    def terminate_individual_agent(self, agent_name):
        if self.level == 'executive':
            for process in self.processes:
                if process._args[0].name == agent_name:
                    process.terminate()
                    print(f"{agent_name} terminated")
                    return
            print(f"Agent {agent_name} not found")

    def track_n_clean(self, request):
        print(f"{self.name} is tracking and cleaning the request: {request}")

    def receive_user_interrupt(self, request):
        print(f"{self.name} received a user interrupt for request: {request}")

    def receive_request(self, request):
        print(f"{self.name} received a request: {request}")

    def analyze_request(self, request):
        print(f"{self.name} analyzed the request: {request}")
        
    def search_content(self, query, top_k=5):
        if self.role == 'Embed':
            return self.embedding_tools.search_based_on_query(query, self.planning_context.related_files, top_k)

    def recommend_content(self, item_key, top_k=5):
        if self.role == 'Embed':
            return self.embedding_tools.recommend_based_on_query(item_key, self.planning_context.related_files, top_k)

    def call_tool_method(self, method_name, *args, **kwargs):
        if self.role == 'Embed':
            if hasattr(self.embedding_tools, method_name):
                method = getattr(self.embedding_tools, method_name)
                return method(*args, **kwargs)
            else:
                raise AttributeError(f"EmbeddingTools does not have a method named '{method_name}'")


    def search_conv_history(self, query,top_k=5):
      return conversation_history.search(query,top_k)

    def search_redis(self,key_pattern='*'):
      return vector_db.get_keys(key_pattern)


if __name__ == '__main__':
    agent_parameters = 'model', 'role', 'level', 'traits', 'tools', 'agent_id'
    executive_agent = Agent(model_data, 'Executive', 'executive', *agent_parameters[2:])


