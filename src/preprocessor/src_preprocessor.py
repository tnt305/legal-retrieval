import pandas as pd
import numpy as np
from tqdm import tqdm



train =  pd.read_csv("./data/train.csv")
train['cid'] = train['cid'].progress_apply(lambda x: x.strip("[]").strip())
train['cid'] = train['cid'].progress_apply(lambda x: ", ".join(x.split()))
train.to_csv("data/corpus_v1.csv", index = False)

