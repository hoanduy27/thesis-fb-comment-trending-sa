s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
def remove_accents(input_str):
	s = ''
	for c in input_str:
		if c in s1:
			s += s0[s1.index(c)]
		else:
			s += c
	return s

def get_word_embedding(word_index , embedding_path):
  import numpy as np
  from gensim.models import KeyedVectors
  w2v = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
  w2v_vocab = w2v.vocab

  vocab_size = min(len(word_index), len(w2v_vocab))
  embedding_dim = w2v.vector_size
  embedding_matrix = np.zeros((vocab_size + 1, embedding_dim))

  for word, i in word_index.items():
    try:
      embedding_vector = word_index[word]
      embedding_matrix[i] = embedding_vector
    except KeyError:
      embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25), 400)

  return vocab_size, embedding_dim, embedding_matrix

