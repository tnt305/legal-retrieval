import json
import os
import pickle
import yaml
from tqdm import tqdm
from datasets import Dataset

def read_json(folder : str):
    with open(folder, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def save_json(data_list: list, folder: str):
    #'./data/update_vocab.json'
    with open(folder, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

def save_pickle(folder: str, data):
    with open(folder, 'wb') as f:
        pickle.dump(data, f)

def read_pickle(folder: str):
    with open(folder, 'rb') as f:
        data = pickle.load(f)
    return data

def read_txt(folder : str):
    with open(folder, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def read_yaml(folder : str):
    with open(folder, 'r') as f:
        data = yaml.safe_load(f)
    return data

def prepare_training_dataset(queries, corpus, relevant_docs):
    anchors = []
    positives = []
    
    # Sử dụng tqdm để theo dõi tiến trình của vòng lặp
    for query_id, docs in tqdm(relevant_docs.items(), desc='Processing queries'):
        for doc_id in docs:
            try:
                # Thử truy cập cả query và document
                anchor = queries[str(query_id)]
                positive = corpus[str(doc_id)]

                # Nếu không gặp lỗi, append vào danh sách
                anchors.append(anchor)
                positives.append(positive)

            except KeyError as e:
                # In ra thông báo lỗi và tiếp tục
                # print(f"Lỗi KeyError: {e} - Bỏ qua query_id: {query_id}, doc_id: {doc_id}")
                continue

    df = {
        "anchor": anchors,
        "positive": positives
    }

    return Dataset.from_dict(df)