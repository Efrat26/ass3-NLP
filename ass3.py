import sys
from time import time
import numpy as np
import pandas as pd

class Word2Vec:
    def __init__(self):
        self.targetWords = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar', 'piano']
        self.words = []
        self.vectors = []
        self.word_to_vec = {}
        self.similar_words = {}

    def load_words_and_vectors(self, file_name):
        #number of columns
        with open(file_name, 'r') as f:
            num_cols = len(f.readline().split())
        num_row = sum(1 for line in open(file_name, encoding='utf-8'))
        cols = num_cols-1
        self.vectors = []
        self.vectors = np.empty([num_row, cols], dtype=np.float32)
        i = 0
        # nplines = np.loadtxt(file_name,)
        with open(file_name, 'r', encoding='utf-8') as f:
            for line in f:

                self.vectors[i] = line.rstrip().split()[1:]
                i += 1
            #print(i)
        #self.vectors = pd.read_csv(file_name, header=None, delimiter=' ', usecols=range(1, num_cols)).values
        self.words = pd.read_csv(file_name, header=None, delimiter=' ', dtype=str, usecols=[0]).values

        print("finishing loadeding words & vecrots")

    def find_sim_word(self):
        for word in self.targetWords:
            sim_words = [word[0] for word in self.calc_sim(self.word_to_vec[word])[1:21]]
            self.similar_words[word] = sim_words
        print("finishing finding similar words")

    def calc_sim(self, word_vec):
        dt = self.vectors.dot(word_vec)  # calc dot-product
        sim = dt.argsort()[-1:10:-1]
        return self.words[sim]  # return list of similar words (each is string)

    def printResults(self):
        output_file = open('word2vec2_results_' + sys.argv[1] + '.txt', 'w', encoding='utf-8')
        for word in self.targetWords:
            output_file.write(word + ":\n")
            word_vec = self.word_to_vec[word]
            output_file.write("similar words:")
            output_file.write(','.join(self.similar_words[word]) + "\n ")

            sim_words = [word[0] for word in self.calc_sim(word_vec)[:11]]

            for sim_word in sim_words:
                output_file.write('\t' + sim_word + '\n')
                output_file.write('\n')

        output_file.close()

    def create_word_to_vec(self):
        w2i = {self.words[i][0]: i for i in range(len(self.words))}

        for word in self.targetWords:
            self.word_to_vec[word] = self.vectors[w2i[word]]

if __name__ == '__main__':
    file_name_words = 'deps.words'
    file_name_contexts = 'deps.contexts'
    if len(sys.argv) > 0:
        file_name_words = sys.argv[1]
        file_name_contexts = sys.argv[2]
    w2v = Word2Vec()
    w2v.load_words_and_vectors(file_name_words)
    w2v.create_word_to_vec()
    w2v.find_sim_word()
    w2v.load_words_and_vectors(file_name_contexts)
    w2v.printResults()
    print("finishing process!")