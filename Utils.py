import os, wget, zipfile, numpy as np, pandas as pd
from ast import literal_eval
from RedisDB import vector_db, text_db, code_db, pdf_db

class Utils:
    def __init__(self, data_path='../../data/', file_name='vector_database_articles_embedded'):
        self.data_path = data_path
        self.file_name = file_name
        self.csv_file_path = os(data_path, file_name + '.csv')

    def download_data(self, data_url, download_path='./'):
        zip_file_path = os.path.join(download_path, self.file_name + '.zip')
        if os.path.isfile(self.csv_file_path):
            print('File already downloaded')
        elif os.path.isfile(zip_file_path):
            print('Zip downloaded but not unzipped, unzipping now...')
        else:
            print('File not found, downloading now...')
            wget.download(data_url, out=download_path, bar=True)

        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)

        os.remove(zip_file_path)
        print(f"File downloaded to {self.data_path}")

    def read_data(self):
        data = pd.read_csv(self.csv_file_path)
        data['title_vector'] = data.title_vector.apply(literal_eval)
        data['content_vector'] = data.content_vector.apply(literal_eval)
        data['vector_id'] = data['vector_id'].apply(str)
        
        for index,row in data.iterrows():
            vector_db.add_item(row["vector_id"], row["content_vector"])

    def save_data_to_redis(self):
        data = pd.read_csv(self.csv_file_path)
        data['title_vector'] = data.title_vector.apply(literal_eval)
        data['content_vector'] = data.content_vector.apply(literal_eval)
        data['vector_id'] = data['vector_id'].apply(str)

        for index, row in data.iterrows():
            vector_db.add_item(row["vector_id"], row["content_vector"])

    def filter_data(self, column_name, value):
        filtered_data_keys = vector_db.get_keys_by_prefix(column_name + ":" + value)
        filtered_data_values = [vector_db.get_item(key) for key in filtered_data_keys]

        return list(zip(filtered_data_keys, filtered_data_values))

    def filter_data_multiple_conditions(self, conditions):
        result_keys = []
        
        for column_name, value in conditions.items():
            condition_keys = vector_db.get_keys_by_prefix(column_name + ":" + value)
            if not result_keys:
                result_keys = condition_keys
            else:
                result_keys = list(set(result_keys).intersection(condition_keys))

        result_values = [vector_db.get_item(key) for key in result_keys]
        
        return list(zip(result_keys, result_values))
        
    def ensure_output_directory_exists():
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

    def write_response_to_file(response, filename):
        with open(os.path.join(OUTPUT_DIRECTORY, filename), 'w', encoding='utf-8') as file:
            file.write(response)

class IngestFiles:
    async def ingest_git_repo(repo_url: str, file_types: list = ['.cs', '.html', '.js', '.py']):
        print('Ingesting Git repository...')
        tmp_dir = tempfile.mkdtemp()
        repo = Repo.clone_from(repo_url, tmp_dir)

        async def process_file(file_path):
            print(f"Processing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            analysis = await get_ada_embeddings(content)
            if analysis:
                code_db.add_item(f"repo:{os.path.relpath(file_path, tmp_dir)}", analysis)
            else:
                print(f"No analysis data for {file_path}")

        tasks = [asyncio.ensure_future(process_file(file_path)) for file_path in glob.glob(tmp_dir + '/**', recursive=True) if os.path.isfile(file_path) and os.path.splitext(file_path)[-1] in file_types]
        await asyncio.gather(*tasks)
        shutil.rmtree(tmp_dir, onerror=remove_readonly)
        print('Ingestion complete.')

    async def ingest_pdf_files(directory: str):
        print("Ingesting PDF files...")
        pdf_files = glob.glob(os.path.join(directory, '*.pdf'))

    async def process_pdf(pdf_file):
        print(f"Processing PDF: {pdf_file}")
        with pdfplumber.open(pdf_file) as pdf:
            content = ''.join(page.extract_text() for page in pdf.pages)
            analysis = await get_ada_embeddings(content)
            if analysis:
                vector_db.add_item(f"pdf:{os.path.basename(pdf_file)}", analysis)
            else:
                print(f"No analysis data for {pdf_file}")
                
        await asyncio.gather(*(process_pdf(pdf_file) for pdf_file in pdf_files))
        print("PDF ingestion complete.")

    async def get_pdf_library() -> str:
        pdf_keys = vector_db.get_keys_by_prefix('pdf:')
        pdf_library = ''
        for key in pdf_keys:
            pdf_name = key.split('pdf:')[-1]
            pdf_content = r.get(key).decode('utf-8')
            pdf_library += f"{pdf_name}:\n{pdf_content}\n"
        return pdf_library

    async def print_files_in_redis_memory():
        print('\nFiles in Redis memory:')
        keys = vector_db.get_keys_by_prefix('pdf:')
        for key in keys:
            pdf_name = key.split('pdf:')[-1]
            print(f"- {pdf_name}")
