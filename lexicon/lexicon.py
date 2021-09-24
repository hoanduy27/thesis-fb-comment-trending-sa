import pandas as pd
import re
import numpy as np
def split_SWN_by_pos(SWN_path, pos, savepath=''):
  ds = pd.read_csv(SWN_path, delimiter='\t', header=21, names=['POS',	'ID', 'PosScore',	'NegScore',	'SynsetTerms',	'Gloss'])
  df = pd.DataFrame(ds, columns=['POS', 'PosScore', 'NegScore', 'SynsetTerms'])
  # pos == '' : get terms from all types of POS
  pos_df = df.loc[(df['POS'] == pos) | (pos=='') ]
  
  sents = np.empty([0,3])
  pattern = re.compile(r'(#([0-9])*)+')

  for id, row in pos_df.iterrows():
    synset = re.sub(pattern, '|', row['SynsetTerms'])
    for sent in synset.strip().split('|')[:-1]:
      sent_norm = sent.replace('_', ' ').strip()
      pos = round(row['PosScore'],3)
      neg = round(row['NegScore'],3)
      sents = np.append(sents, [[sent_norm, pos, neg]], axis=0)

  if savepath != '':
    pd.DataFrame(sents.astype(object), columns=['Term', 'PosScore', 'NegScore']).to_csv(savepath)
  return sents.astype(object)

# import os

# cur_dir = os.path.abspath(__file__)
# cur_dir = os.path.dirname(cur_dir)

# SWN_PATH = f'{cur_dir}/VSWN/ViSentiWordnet_ver1.0.txt'
# SAVE_DIR = f'{cur_dir}/VSWN' 
# for pos in ['v', 'a', 'n', 'r', '']:
#   split_SWN_by_pos(SWN_PATH, pos, f'{SAVE_DIR}/VSW{pos}.csv')
