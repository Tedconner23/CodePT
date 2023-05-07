openai.api_key=os.environ['OPENAI_API_KEY']

OUTPUT_DIRECTORY='C:\\Projects\\GPT Output'
REPO_DIRECTORY='C:\\Projects\\GPT Output'
INGEST_DIRECTORY='C:\\Projects\\GPT Output'
CONVERSATION_HISTORY_KEY = 'conversation_history'

endpoints = {'/v1/chat/completions': ['gpt-4', 'gpt-3.5-turbo'],'/v1/completions': ['text-davinci-003', 'text-davinci-002', 'text-curie-001', 'text-babbage-001', 'text-ada-001'],'/v1/embeddings': ['text-embedding-ada-002']
smart_models={'gpt-4':'Creative, high-quality, versatile, advanced chatbot, deep troubleshooting, extensive context, rich history','text-davinci-003':'Complex tasks, context understanding, detailed responses, code writing, image analysis, math analysis, advanced problem-solving','text-davinci-002':'Quality, versatility, balanced cost, code writing, image analysis, math analysis, intermediate problem-solving'}
standard_models={'gpt-3.5-turbo':'Fast, high-quality, cost-effective, chatbot, troubleshooting, context, history, rapid problem-solving','text-curie-001':'Fast, cost-effective, general tasks, moderate complexity, intent interpretation, data processing, efficient problem-solving'}
quick_models={'text-curie-001':'Fast, cost-effective, general tasks, moderate complexity, intent interpretation, data processing, efficient problem-solving','text-babbage-001':'Quick, simple tasks, lower cost, basic intent interpretation, lightweight data preprocessing, straightforward problem-solving','text-ada-001':'Fastest, basic tasks, lowest cost, minimal complexity, quick response generation, simple problem-solving'}
embed_model={'text-embedding-ada-002':'Embeddings, similarity search, clustering, model interaction, redis, files, scripts, semantic analysis'}
syntax_error_checking_models={'text-curie-001':'Fast, cost-effective, general tasks, moderate complexity, intent interpretation, data processing, efficient problem-solving','text-babbage-001':'Quick, simple tasks, lower cost, basic intent interpretation, lightweight data preprocessing, straightforward problem-solving'}
validation_models={'text-davinci-002':'Quality, versatility, balanced cost, code writing, image analysis, math analysis, intermediate problem-solving','gpt-3.5-turbo':'Fast, high-quality, cost-effective, chatbot, troubleshooting, context, history, rapid problem-solving'}
class_level_validation_models={'text-davinci-003':'Complex tasks, context understanding, detailed responses, code writing, image analysis, math analysis, advanced problem-solving','gpt-3.5-turbo':'Fast, high-quality, cost-effective, chatbot, troubleshooting, context, history, rapid problem-solving'}
full_plan_completion_models={'text-davinci-003':'Complex tasks, context understanding, detailed responses, code writing, image analysis, math analysis, advanced problem-solving'}
code_assemblers={'gpt-3.5-turbo':'Fast, high-quality, cost-effective, chatbot, troubleshooting, context, history, rapid problem-solving','text-curie-001':'Fast, cost-effective, general tasks, moderate complexity, intent interpretation, data processing, efficient problem-solving','text-babbage-001':'Quick, simple tasks, lower cost, basic intent interpretation, lightweight data preprocessing, straightforward problem-solving','text-ada-001':'Fastest, basic tasks, lowest cost, minimal complexity, quick response generation, simple problem-solving'}
model_data={'gpt-4':{'endpoint':'/v1/chat/completions','token_limits':8192,'window_size':8192,'step_size':4096},'gpt-3.5-turbo':{'endpoint':'/v1/chat/completions','token_limits':4096,'window_size':4096,'step_size':2048},'text-davinci-003':{'endpoint':'/v1/completions','token_limits':4096,'window_size':4096,'step_size':2048},'text-davinci-002':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-curie-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-babbage-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-ada-001':{'endpoint':'/v1/completions','token_limits':2048,'window_size':2048,'step_size':1024},'text-embedding-ada-002':{'endpoint':'/v1/embeddings','token_limits':2048,'window_size':2048,'step_size':1024}}