import re, codecs

accents = '/tmp/accents.txt' 
abbrev = '/tmp/abbreviations.txt'

with codecs.open(accents, 'r', encoding='UTF-8') as f:
  accents_dict = {}
  for line in f.readlines():
    k, v = line.split('\t')
    accents_dict[k] = v.replace('\n', '')

with codecs.open(abbrev, 'r', encoding='UTF-8') as f:
  abbrev_dict = {}
  for line in f.readlines():
    k, v = line.split('\t')
    abbrev_dict[k] = v.replace('\n', '')

def normalize_accent(text):
  for k, v in accents_dict.items():
    text = text.replace(k, v)  
  return text 

def normalize_abbrev(text):
  for k, v in abbrev_dict.items():
    text = text.replace(k, v)
  return text

def normalize_repetive(text):
  return re.sub(r'([A-Z])\1+', lambda match: match[1], text, flags=re.IGNORECASE)



def normalize(text):
  text = normalize_abbrev(text)
  text = normalize_accent(text)
  text = normalize_repetive(text)
  return text
