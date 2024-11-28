import gensim

model_path = "../BioWordVec_PubMed_MIMICIII_d200.vec.bin"
model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)
#kijken of het model goed is geladen
len(model.key_to_index) #16545452

#het model positioneert de woorden in een 200-dimensionale vectorruimte
s = model['man']
type(s) #ndarray
s.shape #(200,)