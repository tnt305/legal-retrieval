import random
import os
import pickle
import pandas as pd
from tqdm import tqdm
from text_preprocessor import TextPreprocessing
from multiprocessing import Pool

# Khởi động tqdm cho pandas
tqdm.pandas()

# Đọc dữ liệu từ các tệp CSV
train = pd.read_csv("./../../data/train.csv")
corpora = pd.read_csv("./../../data/corpus.csv")

# Tiền xử lý dữ liệu train
train['cid'] = train['cid'].progress_apply(lambda x: x.strip("[]").strip())
train['cid'] = train['cid'].progress_apply(lambda x: ", ".join(x.split()))
train['context'] = train['context'].progress_apply(lambda x: x.strip("[]").strip("'").strip('"'))
train['context'] = train['context'].progress_apply(lambda x: x.strip('"'))

# Tiền xử lý dữ liệu corpora
corpora['cid'] = corpora['cid'].apply(str)

# Lọc các hàng có len = 1 trong cột 'cid'
train['len'] = train['cid'].apply(lambda x: len(x.split(',')))
train_unique = train[train['len'] == 1]

# Kết hợp context theo cid
merged_train = train_unique.groupby('cid')['context'].agg(' '.join).reset_index()

# Tìm các mục mới chưa có trong corpora và cập nhật
new_entries = merged_train[~merged_train['cid'].isin(corpora['cid'])]
new_entries = new_entries.rename(columns={'context': 'text'})
corpora = pd.concat([corpora, new_entries], ignore_index=True)

# Tạo dictionary cho queries, corpus, và relevant_docs
queries = {str(qid): context for qid, context in zip(train['qid'], train['context'])}
corpus = {str(cid): text for cid, text in zip(corpora['cid'], corpora['text'])}
relevant_docs = {str(i): set(j.split(', ')) for i, j in zip(train['qid'], train['cid'])}

# Lưu các tệp pickle trước khi xử lý
with open('/home/thiendc/projects/legal_retrieval/data/processed/queries_before.pkl', 'wb') as f:
    pickle.dump(queries, f)

with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_before.pkl', 'wb') as f:
    pickle.dump(corpus, f)

with open('/home/thiendc/projects/legal_retrieval/data/processed/relevant_docs.pkl', 'wb') as f:
    pickle.dump(relevant_docs, f)

# Tiền xử lý với TextPreprocessing
preprocessor = TextPreprocessing()

def process_item(item):
    key, value = item
    return key, preprocessor.preprocess_text(value)

# Sử dụng multiprocessing để xử lý đồng thời queries
with Pool(processes=os.cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_item, queries.items()), total=len(queries), desc="Processing queries"))
queries = dict(results)

# Lưu kết quả queries sau khi xử lý
with open('/home/thiendc/projects/legal_retrieval/data/processed/queries_after.pkl', 'wb') as f:
    pickle.dump(queries, f)

# Sử dụng multiprocessing để xử lý đồng thời corpus
with Pool(processes=os.cpu_count()) as pool:
    results = list(tqdm(pool.imap(process_item, corpus.items()), total=len(corpus), desc="Processing corpus"))
corpus = dict(results)

# Lưu kết quả corpus sau khi xử lý
with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_after.pkl', 'wb') as f:
    pickle.dump(corpus, f)
