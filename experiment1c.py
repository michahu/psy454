import torch
from torch import nn
import torch.optim as optim

import torchtext
import pandas as pd
import numpy as np
import sklearn.metrics
from collections import defaultdict
from tqdm import tqdm

from models import MLP1Base, MLP2Base, MetaMLPModel
from torchmeta.modules import MetaLinear

###########################
# HYPERPARAMS

LR = 0.1
###########################

def update_ranks(df, vocab, ranks, model):
    for idx, row in tqdm(df.iterrows()):
        w1, w2, w3 = row['prompt_w1'], row['prompt_w2'], row['prompt_w3']

        if w1 not in vocab.stoi or w2 not in vocab.stoi or w3 not in vocab.stoi:
            # print(w1, w2, w3)
            continue
            
        a = vocab[w1]
        b = vocab[w2]
        c = vocab[w3]
        
        row.pop('prompt_w1')
        row.pop('prompt_w2')
        row.pop('prompt_w3')
        
        d = model(a, b, c)

        sim = sklearn.metrics.pairwise.cosine_similarity(d.reshape(1, -1), vocab.vectors)
        sorted_indices = (-sim).argsort()[0]
        stoi = dict(zip(vocab.itos, sorted_indices))
        
        for key, value in row.items():
            if key not in stoi:
                continue
            idx = stoi[key]
            ranks[idx] += value

vocab = torchtext.vocab.GloVe(name='840B', dim=300)
print('loaded vocab')

sgd_linear_params = torch.load('./models/sgd_linear.pt', map_location=torch.device('cpu'))
sgd_mlp1_params = torch.load('./models/sgd_mlp1.pt', map_location=torch.device('cpu'))
sgd_mlp2_params = torch.load('./models/sgd_mlp2.pt', map_location=torch.device('cpu'))
# meta_linear_params = torch.load('./models/meta_linear.pt', map_location=torch.device('cpu'))
meta_linear_params = torch.load('/n/fs/nlp-myhu/research-code/pytorch-maml/output/jan15_analogy/2021-01-15_091502linear/model.th', map_location=torch.device('cpu'))
meta_mlp1_params = torch.load('/n/fs/nlp-myhu/research-code/pytorch-maml/output/jan15_analogy/2021-01-15_091502mlp1/model.th', map_location=torch.device('cpu'))
meta_mlp2_params = torch.load('/n/fs/nlp-myhu/research-code/pytorch-maml/output/jan15_analogy/2021-01-15_091502mlp2/model.th', map_location=torch.device('cpu'))
# meta_mlp1_params = torch.load('./models/meta_mlp1.pt', map_location=torch.device('cpu'))
# meta_mlp2_params = torch.load('./models/meta_mlp2.pt', map_location=torch.device('cpu'))

print('loaded models')

meta_linear = nn.Linear(in_features=300, out_features=300)
meta_mlp1 = MLP1Base(input_dim=300, hidden_dim=500, output_dim=300)
meta_mlp2 = MLP2Base(input_dim=300, hidden_dim=500, output_dim=300)
sgd_linear = nn.Linear(in_features=900, out_features=300)
sgd_mlp1 = MLP1Base(input_dim=900, hidden_dim=1000, output_dim=300)
sgd_mlp2 = MLP2Base(input_dim=900, hidden_dim=1000, output_dim=300)

sgd_linear.weight.data = sgd_linear_params['linear.weight']
sgd_linear.bias.data = sgd_linear_params['linear.bias']

sgd_mlp1.l1.weight.data = sgd_mlp1_params['l1.weight']
sgd_mlp1.l1.bias.data = sgd_mlp1_params['l1.bias']
sgd_mlp1.l2.weight.data = sgd_mlp1_params['l2.weight']
sgd_mlp1.l2.bias.data = sgd_mlp1_params['l2.bias']

sgd_mlp2.l1.weight.data = sgd_mlp2_params['l1.weight']
sgd_mlp2.l1.bias.data = sgd_mlp2_params['l1.bias']
sgd_mlp2.l2.weight.data = sgd_mlp2_params['l2.weight']
sgd_mlp2.l2.bias.data = sgd_mlp2_params['l2.bias']
sgd_mlp2.l3.weight.data = sgd_mlp2_params['l3.weight']
sgd_mlp2.l3.bias.data = sgd_mlp2_params['l3.bias']

meta_linear.weight.data = meta_linear_params['weight']
meta_linear.bias.data = meta_linear_params['bias']

meta_mlp1.l1.weight.data = meta_mlp1_params['features.layer1.linear.weight']
meta_mlp1.l1.bias.data = meta_mlp1_params['features.layer1.linear.bias']
meta_mlp1.l2.weight.data = meta_mlp1_params['classifier.weight']
meta_mlp1.l2.bias.data = meta_mlp1_params['classifier.bias']

meta_mlp2.l1.weight.data = meta_mlp2_params['features.layer1.linear.weight']
meta_mlp2.l1.bias.data = meta_mlp2_params['features.layer1.linear.bias']
meta_mlp2.l2.weight.data = meta_mlp2_params['features.layer2.linear.weight']
meta_mlp2.l2.bias.data = meta_mlp2_params['features.layer2.linear.bias']
meta_mlp2.l3.weight.data = meta_mlp2_params['classifier.weight']
meta_mlp2.l3.bias.data = meta_mlp2_params['classifier.bias']

sgd_linear.eval()
sgd_mlp1.eval()
sgd_mlp2.eval()

def get_sgd_linear(a, b, c):
    tot = torch.cat((a, b, c), dim=0)
    return sgd_linear(tot).detach().numpy()

def get_sgd_mlp1(a, b, c):
    tot = torch.cat((a, b, c), dim=0)
    return sgd_mlp1(tot).detach().numpy()

def get_sgd_mlp2(a, b, c):
    tot = torch.cat((a, b, c), dim=0)
    return sgd_mlp2(tot).detach().numpy()

mlopt = optim.SGD(meta_linear.parameters(), lr=LR)
mm1opt = optim.SGD(meta_mlp1.parameters(), lr=LR)
mm2opt = optim.SGD(meta_mlp2.parameters(), lr=LR)

mlcache = meta_linear.state_dict()
mm1cache = meta_mlp1.state_dict()
mm2cache = meta_mlp2.state_dict()

criterion = nn.MSELoss()

def get_meta_linear(a, b, c):
    meta_linear.load_state_dict(mlcache)
    mlopt.zero_grad()
    ahat = meta_linear(a)
    loss = criterion(ahat, b)
    loss.backward()
    mlopt.step()
    return meta_linear(c).detach().numpy()

def get_m1(a, b, c):
    meta_mlp1.load_state_dict(mm1cache)
    mm1opt.zero_grad()
    ahat = meta_mlp1(a)
    loss = criterion(ahat, b)
    loss.backward()
    mm1opt.step()
    return meta_mlp1(c).detach().numpy()

def get_m2(a, b, c):
    meta_mlp2.load_state_dict(mm2cache)
    mm2opt.zero_grad()
    ahat = meta_mlp2(a)
    loss = criterion(ahat,b)
    loss.backward()
    mm2opt.step()
    return meta_mlp2(c).detach().numpy()

def cd(a, b, c):
    return c 

def paralleogram(a, b, c):
    return b - a + c

cd_ranks = np.zeros(len(vocab.itos))
p_ranks = np.zeros(len(vocab.itos))
ml_ranks = np.zeros(len(vocab.itos))
mm1_ranks = np.zeros(len(vocab.itos))
mm2_ranks = np.zeros(len(vocab.itos)) 
sl_ranks = np.zeros(len(vocab.itos))
sm1_ranks = np.zeros(len(vocab.itos)) 
sm2_ranks = np.zeros(len(vocab.itos)) 

print('init ranks')

ranks = [cd_ranks, p_ranks, ml_ranks, mm1_ranks, mm2_ranks, sl_ranks, sm1_ranks, sm2_ranks]
funcs = [cd, paralleogram, get_meta_linear, get_m1, get_m2, get_sgd_linear, get_sgd_mlp1, get_sgd_mlp2]

df1=pd.read_csv('./parallelograms-revisited/experiment1_data_completions/experiment1a_counts.csv')
df1=df1.drop(columns=['prompt'])

df2=pd.read_csv('./parallelograms-revisited/experiment1_data_completions/experiment1b_counts.csv')
df2=df2.drop(columns=['prompt'])

df3=pd.read_csv('./parallelograms-revisited/experiment1_data_completions/experiment1c_counts.csv')
df3=df3.drop(columns=['prompt'])

dfs = [df1, df2, df3]

print('loaded dfs')

for df in dfs:
    for rank, get_func in zip(ranks, funcs):
        update_ranks(df, vocab, rank, get_func)

print('saving')

with open('exp1data', 'wb') as f:
    np.save(f, ranks)