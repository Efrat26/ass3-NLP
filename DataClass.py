import sys
from collections import defaultdict

class Data:
    def __init__(self, fileName):
        self.sentences = []
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
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
            self.linesInFile = f.readlines()
        print("finished reading data")
    def countNumOfWords(self):
        sentence = []
        #start = True
        for line in self.linesInFile:
            line = line.rstrip().split('\t')
            if line[0] == '1':
                #if start:
                    #start = False
                if sentence.__len__() > 0:
                    self.sentences.append(sentence)
                sentence = []
                    #continue
            if len(line) != 10:
                continue
            sentence.append(line)
            self.num_of_words[line[2]] +=1
        print('finished counting words & separate to sentences')
    def filterWords(self):
        for key in self.num_of_words:
            if self.num_of_words[key] < self.threshold:
                self.num_of_words.pop(key, None)
    def findCoOccurance(self, type):
        for line in self.sentences:
            if line == '\n':
                continue
            self.findCoOccuranceForSentence(line, type)

    def findCoOccuranceForSentence(self, sentence, type):
        return None

if __name__ == '__main__':
    #create data class
    file_name = 'wikipedia_sample_trees_lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)
    data_object.findCoOccurance(3)

