import sys
from collections import defaultdict

class Data:
    def __init__(self, fileName):
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']'])
        self.threshold = 100
        self.distributional_vectors_type3 = []
        self.features_type3_ind = {}
        self.list_of_features_type3 = []
        self.content_words_to_ind = {}
        self.prepsitions = set(['aboard', 'about', 'above','across','after', 'against', 'ahead of',  'along', 'amid',
                               'amidst',  'among', 'around', 'as', 'as far as', 'as of','aside from', 'at',
                                'athwart', 'atop', 'barring', 'because of', 'before', 'behind',
                                'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by',
                                'by means of', 'circa', 'concernir', 'despite', 'down', 'during', 'except', 'except for',
                                'excluding', 'far from', 'following', 'for', 'from', 'in', 'in accordance with', 'in addition to',
                                'in case of', 'in front of', 'in lieu of', 'in place of', 'in spite of',
                                'including', 'inside', 'instead of', 'into', 'like', 'minus', 'near', 'next to',
                                'notwithstanding', 'of', 'off', 'on', 'on account of', 'on behalf of', 'on top of',
                                'onto', 'opposite', 'out', 'out of', 'outside', 'over', 'past', 'plus', 'prior to',
                                'regarding', 'regardless of', 'save', 'since', 'than', 'through', 'throughout', 'till',
                                'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon',
                                'versus', 'via', 'with', 'with regard to', 'within', 'without'])
        self.readData()
        self.countNumOfWords()
        self.filterWords()
        self.mapContentWordsToInd()


    def readData(self):
        with open(file_name, 'r', encoding="utf8") as f:
            self.linesInFile = f.readlines()
        print("finished reading data")
    def countNumOfWords(self):
        for line in self.linesInFile:
            spltted_line = line.split('\t')
            # skip empty lines or function words
            if len(spltted_line) != 10 or spltted_line[3] not in self.content_words_tags or spltted_line[2]\
                    in self.function_words_lemma_form:
                continue
            self.num_of_words[spltted_line[2]] += 1
        print('finished counting words')
    def filterWords(self):
        keys_to_drop = []
        for key in self.num_of_words:
            if self.num_of_words[key] < self.threshold:
                keys_to_drop.append(key)
        for key in keys_to_drop:
            self.num_of_words.pop(key, None)
        print("finished filtering dictionary")
    def mapContentWordsToInd(self):
        i = 0
        for key in self.num_of_words:
            self.content_words_to_ind[key] = i
            i += 1


    def findCoOccurance(self, type):
        sentence = []
        for line in self.linesInFile:
            if line == '\n':
                self.findCoOccuranceForSentence(sentence, type)
                sentence = []
            else:
                sentence.append(line)


    def findCoOccuranceForSentence(self, sentence, type):
        #the n dimension in matrix of features
        add_feature = False
        splitted_sentence = []
        number_of_words = self.num_of_words.keys()
        #split the words by '\t'
        for word in sentence:
            splitted_sentence.append(word.split('\t'))
        #go over the words, each time a target word is selected
        for splitted_word in splitted_sentence:
            if splitted_word[2] not in self.content_words_to_ind:
                continue
            target_word_id = splitted_word[0]
            #find all other words that are related to the target
            for ind in range(0, len(splitted_sentence)):
                if splitted_sentence[ind][6] == target_word_id:
                    if type == 3:
                        #if the word is a preposition
                        if splitted_sentence[ind][3] == 'IN' and (splitted_sentence[ind][1].lower()
                                                                  in self.prepsitions or
                                                                  splitted_sentence[ind][2].lower() in self.prepsitions):
                            print('preposition word: ' + splitted_sentence[ind][1])
                            #add_feature = True #TODO: add the suitable features
                        # if the word is in the list of the words that we are interested in them:
                        elif splitted_sentence[ind][2] in self.num_of_words:
                            #create features for parent
                            feature_child =  splitted_sentence[ind][1] + ' ' +splitted_sentence[ind][3] + ' ' + 'child'
                            feature_parent = splitted_word[1] + ' ' + splitted_word[3] + ' ' + 'parent'
                            add_feature = True
                        if add_feature:
                            self.addFeatureType3(feature_child, splitted_word)
                            self.addFeatureType3(feature_parent, splitted_sentence[ind])
                            add_feature = False
    def addFeatureType3(self, feature, sentence):
        number_of_words = len(self.num_of_words)
        # if the feature is new - add it and create a vector
        if feature not in self.features_type3_ind:
            # add to list, put the index in the the dictionary & create a vector
            self.list_of_features_type3.append(feature)
            self.features_type3_ind[feature] = len(self.list_of_features_type3) - 1
            # create a zero vecctor, put 1 in the place of the parent word
            # (parent has this feature)
            vector = [0] * number_of_words
            word_feature_added_ind = self.content_words_to_ind[sentence[2]]
            vector[word_feature_added_ind] = 1
            self.distributional_vectors_type3.append(vector)
        # if already exists - then put 1 in the feature vector in the place of the parent word
        else:
            vector_ind = self.features_type3_ind[feature]
            vector = self.distributional_vectors_type3[vector_ind]
            word_feature_added_ind = self.content_words_to_ind[sentence[2]]
            vector[word_feature_added_ind] = 1
            self.distributional_vectors_type3[vector_ind] = vector


if __name__ == '__main__':
    #create data class
    file_name = 'wikipedia_sample_trees_lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)
    data_object.findCoOccurance(3)

'''
   sentence = []
   line_num = 0
#start = True
            line = line.split('\t')
    if line[0] == '1':
        #if start:
            #start = False
        if sentence.__len__() > 0:
            self.sentences_dict[line_num] = sentence
            #self.sentences.append(sentence)
        line_num += 1
        sentence = []
            #continue
             sentence.append(line)
#self.sentences = []
        # self.sentences_dict = {}
'''