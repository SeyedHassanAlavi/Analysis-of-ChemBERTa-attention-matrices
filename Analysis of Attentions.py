import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers import Regex
from transformers import utils
from bertviz import model_view , head_view , neuron_view
from bertviz.neuron_view import show
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from tqdm.auto import tqdm

model = AutoModelForMaskedLM.from_pretrained('DeepChem/ChemBERTa-77M-MLM',output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained('DeepChem/ChemBERTa-77M-MLM')

tokenizer.save_pretrained('/path/to/deepchem')

tokenizer_2 = Tokenizer(
  WordLevel.from_file(
    '/path/to/deepchem/vocab.json',
    unk_token='[UNK]'
))
tokenizer_2.pre_tokenizer = Split(
  pattern = Regex("(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|\+|\\\\\/|:|@|\?|>|>>|\*|\$|\%[0-9A-Fa-f]{2}|[0-9])"),
  behavior= 'isolated'
)

def show_head_view(model, tokenizer, sequence):
  inputs = tokenizer.encode(sequence, return_tensors='pt')
  outputs = model(inputs)
  attention = outputs[-1]
  tokens = tokenizer.convert_ids_to_tokens(inputs[0])
  head_view(attention, tokens)
def show_model_view(model, tokenizer, sequence):
  inputs = tokenizer.encode(sequence, return_tensors='pt')
  outputs = model(inputs)
  attention = outputs[-1]
  tokens = tokenizer.convert_ids_to_tokens(inputs[0])
  model_view(attention, tokens)

def get_cut4(att):
  arr = torch.flatten(att).cpu().numpy()
  kde = gaussian_kde(arr)
  x = torch.linspace(np.min(arr), np.max(arr), 100)
  y = kde(x)
  min_idx = argrelextrema(y, np.less)[0]
  min_peaks = x[min_idx]
  if len(min_peaks != 0):
    return min_peaks[0]
  return np.mean(arr)+np.std(arr)*2

def get_smiles_data(smiles):
  progress_bar = tqdm(range(len(smiles)))
  smiles_data = []
  for s in smiles:
    with torch.no_grad():
      input = tokenizer(s,return_tensors='pt', truncation=True)
      output = model(**input, output_attentions=True)
    labels = tokenizer.decode(input.input_ids[0])[5:-5]
    A = []
    M = []
    for l in range(3):
      layer1 = []
      layer2 = []
      for h in range(12):
        att = output.attentions[l][0][h]
        layer1.append(att)
        layer2.append(att > get_cut4(att))
      A.append(torch.stack(layer1))
      M.append(torch.stack(layer2))
    A = torch.stack(A, dim=0)
    M = torch.stack(M, dim=0)
    smiles_data.append({'tokens':labels , 'A':A, 'M':M})
    progress_bar.update(1)
  return smiles_data

def get_measure_data(index,layer,head,vec2):
    data = smiles_data[index]
    vec1 = data['M'][layer,head][:,1:-1].any(0)
    TP = torch.logical_and(vec1, vec2).sum().item()
    FP = torch.logical_and(vec1, ~vec2).sum().item()
    FN = torch.logical_and(~vec1, vec2).sum().item()
    # TN = torch.logical_and(~vec1, ~vec2).sum().item()

    att = torch.where(data['M'][layer,head]>0, data['A'][layer,head], 0)[:,1:-1]
    idx = []
    for i in range(len(vec2)):
      if(vec2[i]):
        idx.append(i)
    wocc = att[:,idx].sum().item()

    att = torch.where(data['M'][layer,head]>0, data['A'][layer,head], 0)[:,1:-1]
    sum = att.sum().item()

    # return TP, FP, FN, TN, wocc, sum
    return TP, FP, FN, wocc, sum


