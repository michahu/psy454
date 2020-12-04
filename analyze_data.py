import torchtext
import pandas as pd
import numpy as np
import sklearn.metrics
from collections import defaultdict
from tqdm import tqdm

def get_ranks(df, vocab, cd_ranks, p_ranks):
    for idx, row in tqdm(df.iterrows()):
    # for idx, row in df.iterrows():
        w1, w2, w3 = row['prompt_w1'], row['prompt_w2'], row['prompt_w3']
        # print(w1, w2, w3)

        if w1 not in vocab.stoi or w2 not in vocab.stoi or w3 not in vocab.stoi:
            print(w1, w2, w3)
            continue
            
        a = vocab[w1]
        b = vocab[w2]
        c = vocab[w3]
        
        row.pop('prompt_w1')
        row.pop('prompt_w2')
        row.pop('prompt_w3')
        
        # print('popped')

        cd_sim = sklearn.metrics.pairwise.cosine_similarity(c.reshape(1, -1), vocab.vectors)
        cd_sorted_indices = (-cd_sim).argsort()[0]
        cd_stoi = dict(zip(vocab.itos, cd_sorted_indices))

        d = b - a + c 
        p_sim = sklearn.metrics.pairwise.cosine_similarity(d.reshape(1, -1), vocab.vectors)
        p_sorted_indices = (-p_sim).argsort()[0]
        p_stoi = dict(zip(vocab.itos, p_sorted_indices))
        
        # print('constructed stoi')
        
        for key, value in row.items():
            if key not in cd_stoi:
                continue
            cd_idx = cd_stoi[key]
            cd_ranks[cd_idx] += value

            p_idx = p_stoi[key]
            p_ranks[p_idx] += value