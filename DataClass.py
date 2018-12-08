import sys
import numpy as np

class Data:
    def __init__(self, fileName):
        self.sentences = []
        self.num_of_words={}
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']'])
        self.threshold = 100
        self.readData()
        self.countNumOfWords()
        self.filterWords()


    def readData(self):
        with open(file_name, 'r', encoding="utf8") as f:
            self.sentences = f.readlines()
        print("finished reading data")
    def countNumOfWords(self):
        for sentence in self.sentences:
            splitted_sentence = sentence.split('\t')
            if len(splitted_sentence) != 10:
                continue
            splitted_sentence[-1] = splitted_sentence[-1].replace('\n', '')
            for word in splitted_sentence:
                if word in self.num_of_words:
                    self.num_of_words[word] += 1
                else:
                    self.num_of_words[word] = 1
    def filterWords(self):
        for key in self.num_of_words:
            if self.num_of_words[key] < self.threshold:
                self.num_of_words.pop(key, None)

if __name__ == '__main__':
    #create data class
    file_name = 'wikipedia_sample_trees_lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)





