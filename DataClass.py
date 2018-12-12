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
        # distributional vectors where the elements are sets contains the word indices has this feature
        self.dist_vec_type3_set_of_words_inds_per_feature = {}
        #mapping between the feature name to the index in the list of features
        #self.features_type3_ind = {}
        #mapping between feature to vector
        self.distributional_vectors_type3 = {}
        #mapping between each vector to the indcies contained in
        self.dis_vector_word_ind_type3 = {}
        #list of features
        #self.list_of_features_type3 = []
        self.content_words_to_ind = {}
        self.feature_to_index_type3 = {}
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

    '''
    handle case 3.1 where the target word points to a preposition -> need to find the head noun that the
    preposition points to
    '''
    def findNounPrepositionPointsTo(self, prep_word, sentences):
        prep_word_index = int(prep_word[0])
        list_of_nouns = ['NN', 'NNS', 'NNP', 'NNPS']
        root_index = 0
        index = prep_word_index
        stop = False
        # case 3.2 - find the word that the preposition points to and return it with the deprel to the preposition
        current_sentence = sentences[index]
        while not stop:
            current_sentence_tag = current_sentence[3]
            current_sentence_head = int(current_sentence[6])
            if current_sentence[7] == 'ROOT':
                root_index = int(current_sentence[0])-1
            if current_sentence_head == prep_word_index:
                #and current_sentence_tag != 'DT'
                return [current_sentence[1], current_sentence[7]]
            elif index+1 < len(sentences):
                index += 1
                #current_sentence_head = int(current_sentence[6])
                current_sentence = sentences[index]
            else:
                return [sentences[root_index][1], sentences[root_index][7]]
                #return None

    '''
        handle case 3.2 where a preposition points to the target word -> need to find the word that points on that
        preposition and return it with the relation
        '''
    def findWordPointsToPreposition(self, prep_word, sentences):
        prep_word_index = prep_word[0]
        root_index = 0
        index = 0
        stop = False
        # case 3.2 - find the word that the preposition points to and return it with the deprel to the preposition
        current_sentence = sentences[index]
        while not stop:
            current_sentence_tag = current_sentence[3]
            current_sentence_head = current_sentence[6]
            if current_sentence[7] == 'ROOT':
                root_index = int(current_sentence[0]) - 1
            if current_sentence_head == prep_word_index and current_sentence_tag != 'DT':
                return [current_sentence[1], current_sentence[7]]
            elif index+1 < len(sentences):
                index += 1
                #current_sentence_head = int(current_sentence[6])
                current_sentence = sentences[index]
            else:
                return [sentences[root_index][1], sentences[root_index][7]]

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
            #case 3.1: target word points to preposition
            elif daughter_stem in self.prepsitions or daughter_tag == 'IN':
                returned_value = self.findNounPrepositionPointsTo(target_word, splitted_sentences)
                if returned_value != None:
                    feature_child = returned_value[0] + ' ' + target_word[1] + ' ' + daughter_stem + '-' + returned_value[1]
                    self.addFeatureType3(feature_child, target_word)
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
        if feature not in self.feature_to_index_type3:
            self.feature_to_index_type3[feature] = float(len(self.feature_to_index_type3))
            feature_as_index = self.feature_to_index_type3[feature]
            # add to list, put the index in the the dictionary & create a vector
            #self.list_of_features_type3.append(feature)
            #self.features_type3_ind[feature] = len(self.list_of_features_type3) - 1
            # create a zero vecctor, put 1 in the place of the parent word
            # (parent has this feature)
            set_holds_ones_positions = set()
            dist_vector = []
            vector_to_word_map = {}
            if sentence[1] in self.content_words_to_ind:
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            else:
                self.content_words_to_ind[sentence[1]] = len(self.content_words_to_ind)
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            #adding to set the word index
            set_holds_ones_positions.add(word_feature_added_ind)
            #adding a mapping that from the first word in this feature to the index
            vector_to_word_map[word_feature_added_ind] = 0
            #adding 1 for the counter to the word with this context
            dist_vector.append(1)
            #vector[word_feature_added_ind] = 1
            self.dist_vec_type3_set_of_words_inds_per_feature[feature_as_index] = set_holds_ones_positions
            self.distributional_vectors_type3[feature_as_index] = dist_vector
            self.dis_vector_word_ind_type3[feature_as_index] = vector_to_word_map
        # if already exists - then put 1 in the feature vector in the place of the word
        else:
            feature_as_index = self.feature_to_index_type3[feature]
            #gettign the feature index
            #vector_ind = self.features_type3_ind[feature]
            #getting the set of words that include this feature
            set_of_words_with_feature = self.dist_vec_type3_set_of_words_inds_per_feature[feature_as_index]
            #getting the dis vector
            dist_vec = self.distributional_vectors_type3[feature_as_index]
            #getting the mapping from vector ind to word ind
            mapping_from_ind_to_word = self.dis_vector_word_ind_type3[feature_as_index]
            if sentence[1] in self.content_words_to_ind:
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            else:
                self.content_words_to_ind[sentence[1]] = len(self.content_words_to_ind)
                word_feature_added_ind = self.content_words_to_ind[sentence[1]]
            #if word already was with this context
            if word_feature_added_ind in set_of_words_with_feature:
                word_ind_in_vec = mapping_from_ind_to_word[word_feature_added_ind]
                #add 1 to counter
                dist_vec[word_ind_in_vec] += 1
            else:
                #adding mapping of the word
                mapping_from_ind_to_word[word_feature_added_ind] = len(set_of_words_with_feature)
                self.dis_vector_word_ind_type3[feature_as_index] = mapping_from_ind_to_word
                #add word to set
                set_of_words_with_feature.add(word_feature_added_ind)
                self.dist_vec_type3_set_of_words_inds_per_feature[feature_as_index] = set_of_words_with_feature
                #add 1 to dist vecor
                dist_vec.insert(mapping_from_ind_to_word[word_feature_added_ind], 1)
            #vector[word_feature_added_ind] = 1
            self.distributional_vectors_type3[feature_as_index] = dist_vec


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