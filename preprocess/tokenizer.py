import os, sys

cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

class WordSegmenter:
  from pyvi import ViTokenizer
  from vncorenlp import VnCoreNLP as __VnCoreNLP
  VnCoreNLPTokenizer = __VnCoreNLP(f'{cur_dir}/../external/vncorenlp/VnCoreNLP-1.1.1.jar', annotators='wseg', max_heap_size='-Xmx500m')

  def segmentize(segmenter):
    def call(text):
      if segmenter == 'pyvi':
        text = text.strip(' ')
        return WordSegmenter.ViTokenizer.tokenize(text).split(' ') if text else []
      elif segmenter == 'vncorenlp':
        from functools import reduce
        segments = WordSegmenter.VnCoreNLPTokenizer.tokenize(text)
        return list(reduce(lambda lst, seg: lst + seg, segments, []))
    return call

def text_2_bpe_sequences(texts, bpe, vocab, maxlength='auto', truncating=True, padding=True):
  """Tokenize segmented texts to bpe sequences
  Return (vocab size, max length, bpe sequences)"""
  import numpy as np

  EOS = vocab.eos_index
  PAD = vocab.pad_index
  
  # Vectorize functions
  join_word = np.vectorize(' '.join)
  subword_gen = np.vectorize(bpe.encode)
  def bpe_seq_gen(subwords):
    def execute(subword):      
      return np.array(vocab.encode_line('<s> ' + str(subword) + ' </s>', append_eos=False, add_if_not_exist=False), dtype='object')
    return np.vectorize(execute)(subwords)

  # Padding and truncating function
  def pad_truncate(sequences, maxlen):
    def pad(seq):
      seq = np.append(seq, [PAD]*(maxlen - len(seq)))
      return seq.astype('int32')

    def truncate(seq):
      seq = seq[:maxlen]
      seq[-1] = EOS
      return seq.astype('int32')

    def execute(seq):
      if padding and truncating:
        return  pad(seq) if len(seq) <= maxlen else truncate(seq)
        return seq
      elif padding:
        return pad(seq) if len(seq) <= maxlen else seq.astype('int32')
      elif truncating:
        return truncate(seq) if len(seq) > maxlen else seq.astype('int32')
      else:
        return seq.astype('int32')

    # out_seq = np.zeros((len(sequences), maxlen))

    return np.array(list(map(execute, sequences)))

    # return out_seq

  # Mask function
  def mask(sequences, maxlen):
    """Mask token with id == pad_id"""
    masked_sequences = np.zeros((len(sequences), maxlen))
    def execute(seq):
      return np.array([int(tok != PAD) for tok in seq])

    return np.array(list(map(execute, sequences)))

  texts_ = texts.copy()
  texts_ = join_word(texts_)
  subwords = subword_gen(texts_)
  bpe_sequences = bpe_seq_gen(subwords)

  if maxlength == 'auto':
    maxlength = max(len(seq) for seq in bpe_sequences)

  bpe_sequences = pad_truncate(bpe_sequences, maxlength)

  masked_sequences = mask(bpe_sequences, maxlength)

  return maxlength, len(vocab.count), bpe_sequences, masked_sequences
