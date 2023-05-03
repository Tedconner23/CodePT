import os
import wget
import zipfile
import numpy as np
import pandas as pd
from ast import literal_eval

class Utils:
    def __init__(self, data_path='../../data/', file_name="vector_database_articles_embedded"):
        self.data_path = data_path
        self.file_name = file_name
        self.csv_file_path = os.path.join(data_path, file_name + ".csv")

    def download_data(self, data_url, download_path="./"):
        zip_file_path = os.path.join(download_path, self.file_name + ".zip")
        if os.path.isfile(self.csv_file_path):
            print("File already downloaded")
        else:
            if os.path.isfile(zip_file_path):
                print("Zip downloaded but not unzipped, unzipping now...")
            else:
                print("File not found, downloading now...")
                # Download the data
                wget.download(data_url, out=download_path, bar=True)
                # Unzip the data
                with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                    zip_ref.extractall(self.data_path)
                # Remove the zip file
                os.remove(zip_file_path)
                print(f"File downloaded to {self.data_path}")

    def read_data(self):
        data = pd.read_csv(self.csv_file_path)
        # Read vectors from strings back into a list
        data['title_vector'] = data.title_vector.apply(literal_eval)
        data['content_vector'] = data.content_vector.apply(literal_eval)
        # Set vector_id to be a string
        data['vector_id'] = data['vector_id'].apply(str)
        return data

    def save_data(self, data):
        data.to_csv(self.csv_file_path, index=False)
        print(f"Data saved to {self.csv_file_path}")

    def update_data(self, new_data):
        if os.path.isfile(self.csv_file_path):
            current_data = self.read_data()
            updated_data = pd.concat([current_data, new_data], ignore_index=True)
            self.save_data(updated_data)
        else:
            self.save_data(new_data)

    def filter_data(self, column_name, value):
        data = self.read_data()
        filtered_data = data[data[column_name] == value]
        return filtered_data