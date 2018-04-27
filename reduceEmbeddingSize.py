import os
import sys

FILE_PATH = './wiki-news-300d-1M-subword.vec'

def reduceEmbeddingSize(vocabSize):

    new_vec_file = './wiki-news-300d-{0}-subword.vec'.format(vocabSize)

    with open(FILE_PATH, 'rb') as f:
        wordVectors = f.readlines()[:vocabSize + 1]
    f.close()

    with open(new_vec_file, 'wb') as f:
        f.writelines(wordVectors)
    f.close()


if __name__ == '__main__':
    vocabSize = int(sys.argv[1])
    reduceEmbeddingSize(vocabSize)


