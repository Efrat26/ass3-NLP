import sys
from collections import defaultdict
import numpy as np

class Data:
    def __init__(self, fileName):
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']'])
        #self.threshold = 1
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
        #self.mapContentWordsToInd()


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


    def isContentWord(self, word_properties):
        if len(word_properties) != 10:
            return False
        word_tag = word_properties[3]
        word_lemma = word_properties[2]
        if word_tag in self.content_words_tags and word_lemma not in self.function_words_lemma_form:
            return True
        return False


    def findNounPrepositionPointsTo(self, prep_word, sentences):
        prep_head_ind = int(prep_word[6])
        stop = False
        list_of_nouns = ['NN', 'NNS', 'NNP', 'NNPS']
        while not stop:
            examined_sentence = sentences[prep_head_ind-1]
            examined_sentence_tag = examined_sentence[3]
            #if examined_sentence_tag in list_of_nouns:#TODO finish method


    def findWordPointsToPreposition(self, prep_word, sentences):
        prep_word_index = int(prep_word[0])
        index = prep_word_index
        stop = False
        # case 3.2 - find the word that the preposition points to and return it with the deprel to the preposition
        current_sentence = sentences[index]
        while not stop:
            current_sentence_tag = current_sentence[3]
            current_sentence_head = int(current_sentence[6])
            if current_sentence_head == prep_word_index:
                #and current_sentence_tag != 'DT'
                return [current_sentence[1], current_sentence[7]]
            elif index+1 < len(sentences):
                index += 1
                #current_sentence_head = int(current_sentence[6])
                current_sentence = sentences[index]
            else:
                return None
        #return None



    def findCoOccurance(self, type):
        num_of_Sentence = 0
        sentence = []
        for line in self.linesInFile:
            if line == '\n':
                num_of_Sentence += 1
                self.findCoOccuranceForSentence(sentence, type)
                sentence = []
                #print('num of sentence: ' + str(num_of_Sentence))
            else:
                sentence.append(line)



    def findCoOccuranceForSentence(self, sentence, type):
        #the n dimension in matrix of features
        add_feature = False
        splitted_sentences = []
        number_of_words = self.num_of_words.keys()
        #split the words by '\t'
        for word in sentence:
            splitted_sentences.append(word.split('\t'))
        #go over the words, each time a target word is selected
        for target_word in splitted_sentences:
            if not self.isContentWord(target_word):
                continue
            target_word_id = target_word[0]
            target_word_is_daughter_id = target_word[6]
            daughter_stem = splitted_sentences[int(target_word_is_daughter_id)-1][2]
            daughter_tag = splitted_sentences[int(target_word_is_daughter_id)-1][3]
            if daughter_stem in self.content_words_to_ind:
                feature_child = splitted_sentences[int(target_word_is_daughter_id)-1][1] + ' ' + target_word[7] + ' ' + 'is_daughter'
                self.addFeatureType3(feature_child, target_word)
                # TODO: add a case where the word is preposition
            #elif daughter_stem in self.prepsitions or daughter_tag == 'IN':
                #print('dauther is a preposition')

            #find all other words that are related to the target
            for ind in range(0, len(splitted_sentences)):
                current_sentence = splitted_sentences[ind]
                #if there is a word in the sentence that has a dependency relation to the target word
                head = current_sentence[6]
                if head == target_word_id:
                    if type == 3:
                        #if the word is a preposition - case 3.2
                        if current_sentence[3] == 'IN' and (current_sentence[1].lower()
                                                                  in self.prepsitions or
                                                            current_sentence[2].lower() in self.prepsitions):
                            returned_value = self.findWordPointsToPreposition(current_sentence, splitted_sentences)
                            if returned_value is None:
                                #print('case 3.2 - returned value is none!')
                                continue
                            else:
                                feature_parent = returned_value[0] + ' ' + target_word[1] + ' ' + \
                                                 current_sentence[2] + '-' + returned_value[1]
                                add_feature = True
                        # if the word's tag is in the list of the words that we are interested in them:
                        elif self.isContentWord(current_sentence[3]):
                            #create features for parent
                            feature_parent = current_sentence[1] + ' ' + target_word[7] + ' ' + 'is_parent'
                            add_feature = True
                        if add_feature:
                            self.addFeatureType3(feature_parent, target_word)
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
            set_holds_ones_positions = set()
            if sentence[1] in self.content_words_to_ind:
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            else:
                self.content_words_to_ind[sentence[1]] = len(self.content_words_to_ind)
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            set_holds_ones_positions.add(word_feature_added_ind)
            #vector[word_feature_added_ind] = 1
            self.distributional_vectors_type3.append(set_holds_ones_positions)
        # if already exists - then put 1 in the feature vector in the place of the parent word
        else:
            vector_ind = self.features_type3_ind[feature]
            vector = self.distributional_vectors_type3[vector_ind]
            if sentence[1] in self.content_words_to_ind:
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            else:
                self.content_words_to_ind[sentence[1]] = len(self.content_words_to_ind)
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            vector.add(word_feature_added_ind)
            #vector[word_feature_added_ind] = 1
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