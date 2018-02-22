import os
pack=['nltk','sklearn','pandas','textblob']
for i in pack:
    os.system('pip install '+i)
corpus=['punkt','averaged_perceptron_tagger']
for j in corpus:
    nltk.download(j)
