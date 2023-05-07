openai.api_key=os.environ['OPENAI_API_KEY']

OUTPUT_DIRECTORY='C:\\Projects\\GPT Output'
REPO_DIRECTORY='C:\\Projects\\GPT Output'
INGEST_DIRECTORY='C:\\Projects\\GPT Output'
CONVERSATION_HISTORY_KEY = 'conversation_history'

agent_models={'ManagerAgent':['text-babbage-001','gpt-3.5-turbo','gpt-4'],'ProjectManagerAgent':['gpt-3.5-turbo','gpt-4'],'PersonaAgent':['gpt-3.5-turbo','gpt-4'],'LeadAgent':['gpt-4','gpt-3.5-turbo','text-davinci-003'],'MemoryAgent':['gpt-3.5-turbo','text-ada-001'],'GitManAgent':['gpt-3.5-turbo','text-ada-001'],'CodeAssemblyAgent':['gpt-3.5-turbo','text-babbage-001'],'TextAssemblyAgent':['gpt-3.5-turbo','text-babbage-001'],'EmbedAgent':['text-embedding-ada-002'],'TerminalAgent':['text-babbage-001','text-ada-001'],'RelayAgent':['gpt-3.5-turbo','text-babbage-001','text-ada-001'],'BrowseAndFileAgent':['text-babbage_001']}
model_data={'gpt-4':{'endpoint':'/v1/chat/completions','token_limits':8192,'window_size':8192,'step_size':4096},'gpt-3.5-turbo':{'endpoint':'/v1/chat/completions','token_limits':4096,'window_size':4096,'step_size':2048},'text-davinci-003':{'endpoint':'/v1/completions','token_limits':4096,'window_size':4096,'step_size':2048},'text-davinci-002':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-curie-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-babbage-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-ada-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-embedding-ada-002':{'endpoint':'/v1/embeddings','token_limits':2048,'window_size':2048,'step_size':1024}}
