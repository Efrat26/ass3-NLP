import sys
import numpy as np

class Data:
    def __init__(self, fileName):
        self.data = {}#key is the word (as "lemma")
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WRB'])
        self.readData()



    def readData(self):
        with open(self.file_name, "r", encoding="utf8") as lines:
            for line in lines:
                if line != "":
                    splitted_line = line.split()
                    if len(splitted_line) != 10:
                        #print(line)
                        continue
                    if splitted_line[3] in self.content_words_tags:
                        vector = {
                            "ID": splitted_line[0],
                            "FORM": splitted_line[1],
                            "LEMMA": splitted_line[2],
                            "CPOSTAG": splitted_line[3],
                            "POSTAG": splitted_line[4],
                            "FEATS": splitted_line[5],
                            "HEAD": int(splitted_line[6]),
                            "DEPREL": splitted_line[7],
                            "PHEAD": splitted_line[8],
                            "PDEPREL": splitted_line[9],
                        }
                        word_as_lemma = splitted_line[2]
                        if word_as_lemma in self.data and vector not in self.data[word_as_lemma]:
                            self.data[word_as_lemma].append(vector)
                        elif word_as_lemma not in self.data:
                            s = []
                            s.append(vector)
                            self.data[word_as_lemma] = s
        print("finished reading data")
if __name__ == '__main__':
    #create data class
    file_name = 'wikipedia.sample.trees.lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)




