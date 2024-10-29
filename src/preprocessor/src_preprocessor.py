import random
import os
import pickle
import pandas as pd
from tqdm import tqdm
from preprocessor.preprocessor import TextPreprocessing
from multiprocessing import Pool

# # Khởi động tqdm cho pandas
# tqdm.pandas()

# # Đọc dữ liệu từ các tệp CSV
# train = pd.read_csv("./../../data/train.csv")
# corpora = pd.read_csv("./../../data/corpus.csv")

# # Tiền xử lý dữ liệu train
# train['cid'] = train['cid'].progress_apply(lambda x: x.strip("[]").strip())
# train['cid'] = train['cid'].progress_apply(lambda x: ", ".join(x.split()))
# train['context'] = train['context'].progress_apply(lambda x: x.strip("[]").strip("'").strip('"'))
# train['context'] = train['context'].progress_apply(lambda x: x.strip('"'))

# # Tiền xử lý dữ liệu corpora
# corpora['cid'] = corpora['cid'].apply(str)

# # Lọc các hàng có len = 1 trong cột 'cid'
# train['len'] = train['cid'].apply(lambda x: len(x.split(',')))
# train_unique = train[train['len'] == 1]

# # Kết hợp context theo cid
# merged_train = train_unique.groupby('cid')['context'].agg(' '.join).reset_index()

# # Tìm các mục mới chưa có trong corpora và cập nhật
# new_entries = merged_train[~merged_train['cid'].isin(corpora['cid'])]
# new_entries = new_entries.rename(columns={'context': 'text'})
# corpora = pd.concat([corpora, new_entries], ignore_index=True)

# # Tạo dictionary cho queries, corpus, và relevant_docs
# queries = {str(qid): context for qid, context in zip(train['qid'], train['context'])}
# corpus = {str(cid): text for cid, text in zip(corpora['cid'], corpora['text'])}
# relevant_docs = {str(i): set(j.split(', ')) for i, j in zip(train['qid'], train['cid'])}

# Lưu các tệp pickle trước khi xử lý
# with open('/home/thiendc/projects/legal_retrieval/data/processed/queries_before.pkl', 'wb') as f:
#     pickle.dump(queries, f)

# with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_before.pkl', 'wb') as f:
#     pickle.dump(corpus, f)

# with open('/home/thiendc/projects/legal_retrieval/data/processed/relevant_docs.pkl', 'wb') as f:
#     pickle.dump(relevant_docs, f)

# Tiền xử lý với TextPreprocessing
preprocessor = TextPreprocessing()
# with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_before.pkl', 'wb') as f:
#     pickle.dump(corpus, f)
def process_item(item):
    key, value = item
    return key, preprocessor.preprocess_text(value)

# # Sử dụng multiprocessing để xử lý đồng thời queries
# with Pool(processes=os.cpu_count()) as pool:
#     results = list(tqdm(pool.imap(process_item, queries.items()), total=len(queries), desc="Processing queries"))
# queries = dict(results)

# Lưu kết quả queries sau khi xử lý
with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_20000.pkl', 'rb') as f:
    corpus = pickle.load(f)

# Sử dụng multiprocessing để xử lý đồng thời corpus
# Sử dụng multiprocessing để xử lý đồng thời queries
with Pool(processes= 40) as pool:
    results = list(tqdm(pool.imap(process_item, corpus.items()), total=len(corpus), desc="Processing queries"))
corpus = dict(results)

# Lưu kết quả queries sau khi xử lý
with open('/home/thiendc/projects/legal_retrieval/data/processed/corpus_after_20000.pkl', 'wb') as f:
    pickle.dump(corpus, f)







# from multiprocessing import Pool, cpu_count
# from functools import partial
# import pickle
# from tqdm import tqdm
# import os
# import time

# def process_batch(batch_index_and_items):
#     batch_index, batch_items = batch_index_and_items
#     results = {}
#     for key, value in batch_items:
#         results[key] = preprocessor.preprocess_text(value)
#     return batch_index, results

# def chunks(data, size):
#     data_items = list(data.items())
#     for i in range(0, len(data_items), size):
#         yield i // size, data_items[i:i + size]

# def save_batch(batch_index, batch_result, output_dir):
#     batch_file = os.path.join(output_dir, f'corpus_batch_{batch_index}.pkl')
#     with open(batch_file, 'wb') as f:
#         pickle.dump(batch_result, f, protocol=pickle.HIGHEST_PROTOCOL)
#     return batch_file

# # Tạo thư mục output nếu chưa tồn tại
# output_dir = '/home/thiendc/projects/legal_retrieval/data/processed/corpus_batches'
# os.makedirs(output_dir, exist_ok=True)

# # Xác định batch size phù hợp
# BATCH_SIZE = 2048
# n_cores = cpu_count()

# # Chia corpus thành các batch nhỏ hơn
# batches = list(chunks(corpus, BATCH_SIZE))

# # Tạo file index để lưu thông tin về các batch
# batch_index = {
#     'created_time': time.time(),
#     'total_batches': len(batches),
#     'batch_files': []
# }

# # Xử lý và lưu từng batch
# with Pool(processes=n_cores) as pool:
#     for batch_index, batch_result in tqdm(
#         pool.imap(process_batch, batches),
#         total=len(batches),
#         desc="Processing and saving batches"
#     ):
#         # Lưu batch ngay khi xử lý xong
#         batch_file = save_batch(batch_index, batch_result, output_dir)
#         batch_index['batch_files'].append(batch_file)
        
#         # Cập nhật và lưu file index sau mỗi batch
#         index_file = os.path.join(output_dir, 'batch_index.pkl')
#         print(f'File được save ở batch {batch_index}')
#         with open(index_file, 'wb') as f:
#             pickle.dump(batch_index, f, protocol=pickle.HIGHEST_PROTOCOL)

# def load_full_corpus(index_file):
#     with open(index_file, 'rb') as f:
#         batch_index = pickle.load(f)
    
#     full_corpus = {}
#     for batch_file in tqdm(batch_index['batch_files'], desc="Loading batches"):
#         with open(batch_file, 'rb') as f:
#             batch_data = pickle.load(f)
#             full_corpus.update(batch_data)
    
#     return full_corpus
