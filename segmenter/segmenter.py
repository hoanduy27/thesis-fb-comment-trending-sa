import os, sys

cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

class WordSegmenter:
  from pyvi import ViTokenizer
  from vncorenlp import VnCoreNLP as __VnCoreNLP
  VnCoreNLPTokenizer = __VnCoreNLP(f'{cur_dir}/vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')

  def tokenizer(segmenter):
    def tokenize(text):
      if segmenter == 'pyvi':
        text = text.strip(' ')
        return WordSegmenter.ViTokenizer.tokenize(text).split(' ') if text else []
      elif segmenter == 'vncorenlp':
        from functools import reduce
        segments = WordSegmenter.VnCoreNLPTokenizer.tokenize(text)
        return list(reduce(lambda lst, seg: lst + seg, segments, []))
    return tokenize
