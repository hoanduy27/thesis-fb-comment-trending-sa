# -*- coding: utf-8 -*-
import re, codecs
import os
import unicodedata
import string
from pyvi import ViTokenizer

REGEX_SPECIAL = '*^$*+?!#|\\()[]'

cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

accents = f'{cur_dir}/map/accents.txt'
syllables = f'{cur_dir}/map/syllables.txt'
abbrev = f'{cur_dir}/map/abbreviations.txt'
url = f'{cur_dir}/map/url.txt'

with codecs.open(accents, 'r', encoding='UTF-8') as f:
  accents_dict = {}
  for line in f.readlines():
    k, v = line.split('\t')
    accents_dict[k] = v.replace('\n', '')

with codecs.open(syllables, 'r', encoding='UTF-8') as f:
  syllables_dict = {}
  for line in f.readlines():
    k, v = line.split('\t')
    syllables_dict[k] = v.replace('\n', '')

with codecs.open(abbrev, 'r', encoding='UTF-8') as f:
  abbrev_dict = {}
  for line in f.readlines():
    k, v = line.split('\t')
    abbrev_dict[k] = v.replace('\n', '')

with codecs.open(url, 'r', encoding='UTF-8') as f:
  re_urls = []
  for url in f.readlines():
    re_urls += [re.compile(url.strip('\n'))]

def remove_html(text):
  return re.sub(r'<[^>]*>', '', text)
 
def convert_utf8(text):
  return unicodedata.normalize('NFC', text)


def normalize_accent(text):
  """a', o` -> á, ò"""
  for k, v in accents_dict.items():
    text = text.replace(k, v)  
  return text 

def normalize_syllable(text):
  """òa, òe, úy -> oà, oè, uý"""
  for k, v in syllables_dict.items():
    text = text.replace(k, v)  
  return text 

def normalize_abbrev(text):
  for k, v in abbrev_dict.items():
    #text = text.replace(k, v)
    k = ''.join('\\'+c if REGEX_SPECIAL.find(c) > -1 else c for c in k)
    r = re.compile(fr'\b{k}\b', flags=re.IGNORECASE)
    text = r.sub(v, text)
  return text

def normalize_repetive(text):
  return re.sub(r'([A-Z])\1+', lambda match: match[1], text, flags=re.IGNORECASE)

def remove_url(text):
  for re_url in re_urls:
    text = re_url.sub('', text) 
  return text

def normalize(text):
  text = remove_html(text)
  text = remove_url(text)
  text = convert_utf8(text)
  text = normalize_accent(text)
  text = normalize_abbrev(text)
  text = normalize_syllable(text)
  text = normalize_repetive(text)

  # Normalize some well-known emojis
  text = re.sub(r'[:=]([\)\]}])\1+', '🙂', text)
  text = re.sub(r'[:=]([\(\[{])\1+', '🙁', text)
  text = re.sub(r'[:=]([vV])+', '🤣', text)
  
  # Remove punctuation
  punc = ''.join([p for p in string.punctuation if p not in '.!?,'])
  translator = str.maketrans(punc, ' '*len(punc))
  text = text.translate(translator)
  text = text.translate(str.maketrans('.!?', '.'*3))

  # Word segmentation
  text = ViTokenizer.tokenize(text, )
  
  return text


# Testing
texts = [
  "ko",
  "t/k/cần bik bạn là ai. Bạn không nổ địa chỉ tại abc@gmail.com hay google.com thì chết cm mày với t! Cho mày 2 ngày",
  "Nhất cô Hằng lun, 5* cô ơi :))",
  "vẽ họa tiết",
  "ca^u â'm co^ chie^u", 
  "   la'o    nga    la'o ngáo. hoa' điên ho'a khùng",
  r":v :vvv :VVV :))) :}}}} :) :((( ={{{",
  "Chắc đc đó. Hôm qua tao có nói với nó rồi mà",
  "ngoooonnnnn",
  '=]]]]]]',
  '<p class=h> abc </p>'
]
decomposed_texts = [unicodedata.normalize('NFD', t) for t in texts]

normalized_texts = [normalize(t) for t in texts]
normalized_texts_decomposed = [normalize(t) for t in decomposed_texts]

for t in normalized_texts:
  print(t)
print(texts == decomposed_texts)
print(normalized_texts == normalized_texts_decomposed)
