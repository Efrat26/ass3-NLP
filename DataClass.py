import sys
from collections import defaultdict
import numpy as np

class Data:
    def __init__(self, fileName):
        self.temp = 0
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']'])
        #self.threshold = 1
        self.threshold = 100
        ### for type 3 dist vectors:
        self.word_to_features_mapping_dict_type3 = {}#key: word, values: features set
        self.features_to_counters_mapping_type3 = {}#key:feature, value: list with counter for each word has this feature
        self.features_set = set()
        self.vector_index_to_word_mapping_dict = {}#key: word as lemma to index (0,1,2...)
        self.word_to_index_mapping = {}




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
        self.indexWords()
        #self.mapContentWordsToInd()


    def readData(self):
        with open(file_name, 'r', encoding="utf8") as f:
            self.linesInFile = f.readlines()
        print("finished reading data")


    def indexWords(self):
        counter = 0
        for key in self.num_of_words:
            self.word_to_index_mapping[key] = counter
            counter += 1
        print('finished indexing words')



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
        if word_properties[2] in self.num_of_words:
            return True
        else:
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
                return [current_sentence[2], current_sentence[7]]
            elif index+1 < len(sentences):
                index += 1
                #current_sentence_head = int(current_sentence[6])
                current_sentence = sentences[index]
            else:
                return [sentences[root_index][2], sentences[root_index][7]]
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
                return [current_sentence[2], current_sentence[7]]
            elif index+1 < len(sentences):
                index += 1
                #current_sentence_head = int(current_sentence[6])
                current_sentence = sentences[index]
            else:
                return [sentences[root_index][2], sentences[root_index][7]]

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
        print('finished')



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
            if daughter_stem is self.isContentWord(splitted_sentences[int(target_word_is_daughter_id)-1]):
                feature_child = splitted_sentences[int(target_word_is_daughter_id)-1][2] + ' ' + target_word[7] + ' ' + 'is_daughter'
                self.addFeatureType3(feature_child, target_word)
            #case 3.1: target word points to preposition
            elif daughter_stem in self.prepsitions or daughter_tag == 'IN':
                returned_value = self.findNounPrepositionPointsTo(target_word, splitted_sentences)
                if returned_value != None:
                    feature_child = returned_value[0] + ' ' + target_word[2] + ' ' + daughter_stem + '-' + returned_value[1]
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
                                feature_parent = returned_value[0] + ' ' + target_word[2] + ' ' + \
                                                 current_sentence[2] + '-' + returned_value[1]
                                add_feature = True
                        # if the word's tag is in the list of the words that we are interested in them:
                        elif self.isContentWord(current_sentence[3]):
                            #create features for parent
                            feature_parent = current_sentence[2] + ' ' + target_word[7] + ' ' + 'is_parent'
                            add_feature = True
                        if add_feature:
                            self.addFeatureType3(feature_parent, target_word)
                            add_feature = False


    def addFeatureType3(self, feature, sentence):
        #check if we have the feature already or not
        target_word_as_lemma = sentence[2]
        if feature in self.features_set:
            self.checkIfTargetWordHasFeature(target_word_as_lemma, feature)

        #we don't have the feature
        else:
            self.checkIfTargetWordHasFeature(target_word_as_lemma, feature)
            #create new dist vect
            new_vec = []
            #add 1 to the counter for the target word
            new_vec.append(1)



    def checkIfTargetWordHasFeature(self, target_word_as_lemma, feature):
        has_feature = False

        if target_word_as_lemma in self.word_to_features_mapping_dict_type3:
            if feature in self.word_to_features_mapping_dict_type3[target_word_as_lemma]:
                has_feature = True
            self.word_to_features_mapping_dict_type3[target_word_as_lemma].add(feature)
        else:
            features = set()
            features.add(feature)
            self.word_to_features_mapping_dict_type3[target_word_as_lemma] = features
        return has_feature


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