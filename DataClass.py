import sys
from collections import defaultdict
import numpy as np
import math
import heapq
class Feature:
    def __init__(self, index):
        self.wordIndex = index
        self.count = 1

class Data:
    def __init__(self, fileName, threshold_value, type_co_occ):
        self.type = type_co_occ
        self.temp = 0
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']', 'either', 'or', 'and', 'moreover'])
        self.threshold = threshold_value
        #self.threshold = 600
        self.word_to_index_mapping = {}
        ##simple start
        ### for type 3 dist vectors:
        self.word_to_dist_vec = {}
        self.word_to_set_of_features = {}
        self.feature_to_set_of_words = {}

        #try to make it quicker
        self.word_to_feature_to_index_dict = {}
        self.word_to_index_to_feature_dict = {}

        ###for PMI values
        self.pmi_word = {}
        self.pmi_att = {}
        self.pmi_word_att = {}
        self.total_co_occ = 0
        # to use less memory PMI vectors include only the values and not a pair of <feature, value>. the order
        #of the features is the same as in the distributional vectors
        self.word_to_pmi_vec = {}


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
        print("number of content words after filtering is: " + str(len(self.num_of_words)))
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

    def findCoOccurance(self):
        num_of_Sentence = 0
        sentence = []
        for line in self.linesInFile:
            if line == '\n':
                num_of_Sentence += 1
                self.findCoOccuranceForSentence(sentence)
                sentence = []
                #print('num of sentence: ' + str(num_of_Sentence))
            else:
                sentence.append(line)
        print('finished')
        print('number of features is: ' + str(len(self.feature_to_set_of_words)))

    '''
    handle case 3.1 where the target word points to a preposition -> need to find the head noun that the
    preposition points to
    '''
    def findNounPrepositionPointsTo(self, prep_word, sentences):
        prep_word_index = int(prep_word[0])
        root_index = 0
        index = prep_word_index
        stop = False
        # case 3.1 - find the word that the preposition points to and return it with the deprel to the preposition
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

    def findCoOccuranceForSentence(self, sentence):
        #the n dimension in matrix of features
        add_feature = False
        splitted_sentences = []
        number_of_words = self.num_of_words.keys()
        #split the words by '\t'
        for word in sentence:
            splitted_sentences.append(word.split('\t'))
        #for type 1 : get the indecies of the content words
        if self.type==1:
            content_words_inds_set = self.getIndOfContentWordsInSentence(splitted_sentences)
        #go over the words, each time a target word is selected
        for i in range(0, len(splitted_sentences)):
            target_word = splitted_sentences[i]
            if not self.isContentWord(target_word):
                continue
            if self.type == 3:
                target_word_id = target_word[0]
                target_word_head_id = target_word[6]
                child_lemma = splitted_sentences[int(target_word_head_id) - 1][2]
                child_tag = splitted_sentences[int(target_word_head_id) - 1][3]
                if self.isContentWord(splitted_sentences[int(target_word_head_id) - 1]):
                    feature_child = splitted_sentences[int(target_word_head_id) - 1][2] + ' ' + 'child'
                    self.addFeature(feature_child, target_word)
                # case 3.1: target word points to preposition
                elif child_lemma.lower() in self.prepsitions or child_tag == 'IN':
                    returned_value = self.findNounPrepositionPointsTo(target_word, splitted_sentences)
                    if returned_value != None:
                        feature_child = returned_value[0] + ' ' + child_lemma + '-' + returned_value[1]
                        self.addFeature(feature_child, target_word)
                # find all other words that are related to the target
                for j in range(0, len(splitted_sentences)):
                    # we don't want to have the target word associated with the target word
                    if i == j:
                        continue
                    current_sentence = splitted_sentences[j]
                    # if there is a word in the sentence that has a dependency relation to the target word
                    head = current_sentence[6]
                    if head == target_word_id:
                        # if the word is a preposition - case 3.2
                        if current_sentence[3] == 'IN' and (current_sentence[1].lower()
                                                            in self.prepsitions or
                                                            current_sentence[2].lower() in self.prepsitions):
                            returned_value = self.findWordPointsToPreposition(current_sentence, splitted_sentences)
                            if returned_value is None:
                                # print('case 3.2 - returned value is none!')
                                continue
                            else:
                                feature_parent = returned_value[0] + ' ' + \
                                                 current_sentence[2] + '-' + returned_value[1]
                                add_feature = True
                        # if the word's tag is in the list of the words that we are interested in them:
                        elif self.isContentWord(current_sentence):
                            # create features for parent
                            feature_parent = current_sentence[2] + ' ' + 'parent'
                            add_feature = True
                        if add_feature:
                            self.addFeature(feature_parent, target_word)
                            add_feature = False
            elif self.type == 2:
                if i > 0:
                    word_before = splitted_sentences[i-1][2]
                    if self.isContentWord(splitted_sentences[i - 1]):
                        self.addFeature(word_before, target_word)
                if i < len(splitted_sentences)-1:
                    word_after = splitted_sentences[i+1][2]
                    if self.isContentWord(splitted_sentences[i+1]):
                        self.addFeature(word_after, target_word)

            elif self.type == 1:
                for index in content_words_inds_set:
                    if index != i:
                        feature = splitted_sentences[index][2]
                        self.addFeature(feature, target_word)


    def getIndOfContentWordsInSentence(self, sentences):
        result = set()
        for i in range(0,len(sentences)):
            if self.isContentWord(sentences[i]):
                result.add(i)
        return result

    def addFeature(self, feature, sentence):
        word_lemma = sentence[2]
        #map feature to the words contained it
        if feature in self.feature_to_set_of_words:
            set_of_words = self.feature_to_set_of_words[feature]
            set_of_words.add(word_lemma)
            self.feature_to_set_of_words[feature] = set_of_words
        else:
            set_of_words = set()
            set_of_words.add(word_lemma)
            self.feature_to_set_of_words[feature] = set_of_words

        #map the word to the distribtional vectors

        if word_lemma in self.word_to_dist_vec:#word lemma was seen before
            dist_vec = self.word_to_dist_vec[word_lemma]
            set_of_features = self.word_to_set_of_features[word_lemma]
            feature_index_dict = self.word_to_feature_to_index_dict[word_lemma]
            index_to_feature_dict = self.word_to_index_to_feature_dict[word_lemma]
            if feature in set_of_features:
                #retrive the index of the feature in the dist_vec
                word_index = feature_index_dict[feature]
                dist_vec[word_index] += 1
                self.word_to_dist_vec[word_lemma] = dist_vec
                #set_of_features.add(feature)
                #self.word_to_set_of_features[word_lemma] = set_of_features
                return
                #for i in range(0,len(dist_vec)):
                 #   current_pair_feature_counter = dist_vec[i]
                  #  if feature == current_pair_feature_counter[0]:

            else:#feature is new
            #if for ended and the feature wasn't found, means we need to create a new pair for it
                set_of_features.add(feature)
                self.word_to_set_of_features[word_lemma] = set_of_features
                new_feature_counter_pair = 1
                dist_vec.append(new_feature_counter_pair)
                self.word_to_dist_vec[word_lemma] = dist_vec
                #add the index of the feature
                feature_index_dict[feature] = len(set_of_features) - 1
                index_to_feature_dict[len(set_of_features) - 1] = feature
                self.word_to_feature_to_index_dict[word_lemma] = feature_index_dict
                self.word_to_index_to_feature_dict[word_lemma] = index_to_feature_dict
                return
        #word not in dictionary - need to add it
        else:
            dist_vec = []
            new_feature_counter_pair = 1
            dist_vec.append(new_feature_counter_pair)
            self.word_to_dist_vec[word_lemma] = dist_vec
            set_of_features = set()
            set_of_features.add(feature)
            self.word_to_set_of_features[word_lemma] = set_of_features

            map_feature_to_index = {}
            index_feature_dict = {}
            map_feature_to_index[feature] = 0
            index_feature_dict[0] = feature
            self.word_to_feature_to_index_dict[word_lemma] = map_feature_to_index
            self.word_to_index_to_feature_dict[word_lemma] = index_feature_dict

            return


    def printHighestFeatures(self, target_words_list):
        result = {}
        for target_word in target_words_list:
            top_20_features = []
            features_names = []
            if target_word in self.word_to_pmi_vec:
                index_dictionary = self.word_to_index_to_feature_dict[target_word]
                features_names = [None]*len(index_dictionary)
                for key in index_dictionary:
                    features_names[key] = index_dictionary[key]
                pmi_vec = self.word_to_pmi_vec[target_word]
                for i in range(0,20):
                    max_index = pmi_vec.index(max(pmi_vec))
                    top_20_features.append(features_names[max_index])
                    pmi_vec.pop(max_index)
                    features_names.pop(max_index)
                result[target_word] =  top_20_features
        #print results
        for target_word in result:
            list_of_words = result[target_word]
            print("top 20 features of " + target_word + " are:" + ', '.join(list_of_words))




    def filterFeatures(self):
        print('starting filtering features')
        if self.type == 1:
            threshold_co_occ = 20
        #else:
        #    thresholf_co_occ = 20
        for word in self.word_to_dist_vec:
            new_dist_vec = []
            dist_vec = self.word_to_dist_vec[word]
            set_features_for_word = self.word_to_set_of_features[word]
            original_dict_index_to_feature = self.word_to_index_to_feature_dict[word]
            #original_dict_feature_to_index = self.word_to_index_to_feature_dict_type3[word]
            dict_index_to_feature = {}
            dict_feature_to_index = {}

            for i in range(0, len(dist_vec)):
                feature = original_dict_index_to_feature[i]
                if dist_vec[i] > threshold_co_occ:
                    index_in_array = len(new_dist_vec)
                    new_dist_vec.append(dist_vec[i])
                    #to know which feature was there
                    dict_index_to_feature[index_in_array] = feature
                    dict_feature_to_index[feature] = index_in_array
                else:
                    #need to remove the feature

                    #remove word from the set of words of the feature
                    original_set_of_words = self.feature_to_set_of_words[feature]
                    original_set_of_words.remove(word)
                    self.feature_to_set_of_words[feature] = original_set_of_words
                    #remove feature from the set of features for word
                    set_features_for_word.remove(feature)
            self.word_to_dist_vec[word] = new_dist_vec
            self.word_to_set_of_features[word] = set_features_for_word
            self.word_to_index_to_feature_dict[word] = dict_index_to_feature
            self.word_to_feature_to_index_dict[word] = dict_feature_to_index
        print('finished filtering features')
        print("number of features after filtering is: " + str(len(self.feature_to_set_of_words)))
        print("number of words after filtering is: " + str(len(self.word_to_dist_vec)))

    def createPMIvectors(self):
        if self.type == 1:
            self.filterFeatures()
        #calculate #(*,*)
        number_of_co_occ_observed_in_corpus = 0
        for key in self.word_to_dist_vec:
            list_of_dist_vecs_for_word = self.word_to_dist_vec[key]
            for i in range(0, len(list_of_dist_vecs_for_word)):
                number_of_co_occ_observed_in_corpus += list_of_dist_vecs_for_word[i]
        #number_of_co_occ_observed_in_corpus *= 2
        if number_of_co_occ_observed_in_corpus == 0:
            print("number of co-occurances is zero!")
            return
        else:
            print("total co-occ is " + str(number_of_co_occ_observed_in_corpus))
            self.total_co_occ = number_of_co_occ_observed_in_corpus

        for feature in self.feature_to_set_of_words:
            list_of_words_has_feature = self.feature_to_set_of_words[feature]
            # print(list_of_words_has_feature)
            #print("feature is: " + feature)
            #print("set of words with this feature: " + ' '.join(list_of_words_has_feature))
            counter = 0
            for word_with_feature in list_of_words_has_feature:
                #  print("word with feature is: " + word_with_feature)
                word_to_features_index_dict = self.word_to_feature_to_index_dict[word_with_feature]
                index = word_to_features_index_dict[feature]
                # print("index is: " + str(index))
                dist_vec = self.word_to_dist_vec[word_with_feature]
                counter += dist_vec[index]
            self.pmi_att[feature] = counter / number_of_co_occ_observed_in_corpus

        # sanity check
        sanity_check_sum_of_features_probabilities = 0
        for feature in self.pmi_att:
            sanity_check_sum_of_features_probabilities += self.pmi_att[feature]
        print("sainty check: sum of probabilities for words " + str(sanity_check_sum_of_features_probabilities))

        # calculate p(word)
        for word in self.word_to_set_of_features:
            list_of_dist_vecs_for_word = self.word_to_dist_vec[word]
            counter = 0
            for i in range(0,len(list_of_dist_vecs_for_word)):
                counter += list_of_dist_vecs_for_word[i]
            self.pmi_word[word] = counter / number_of_co_occ_observed_in_corpus

        #sainty check
        sanity_check_sum_of_words_probabilities = 0
        for word in self.pmi_word:
            sanity_check_sum_of_words_probabilities += self.pmi_word[word]
        print("sanity check: sum of probabilities for words " + str(sanity_check_sum_of_words_probabilities))
        counter = 0
        #calculate p(word,att) & pmi vector (includes the sanity check counter)
        for word in self.word_to_dist_vec:
            pmi_vec = []
            dist_vec = self.word_to_dist_vec[word]
            for att in dist_vec:
                p_att_word = att / number_of_co_occ_observed_in_corpus
                pair = p_att_word
                pmi_vec.append(pair)
                counter += pair
            self.pmi_word_att[word] = pmi_vec
        #sanity check
        print("sanity check for sigma on word,att result is: " + str(counter))

        #create PMI vector
        for word in self.pmi_word_att:
            caculated_pmi_pair = []
            pmi_vec = self.pmi_word_att[word]
            index_to_feature_dict = self.word_to_index_to_feature_dict[word]
            for i in range(0, len(pmi_vec)):
                current_pair = pmi_vec[i]
                feature = index_to_feature_dict[i]
                p_att = self.pmi_att[feature]
                p_word = self.pmi_word[word]
                p_att_word = current_pair
                if p_att == 0 or p_word == 0:
                    result = 0.0
                elif p_att_word == 0:
                    result = 0.0
                else:
                    temp1 = p_att_word / (p_word * p_att)
                    temp2 = math.log(temp1)
                    #print(str(temp2))
                    new_temp2 = format(temp2, '.3f')
                    #new_temp2 = format(temp2, '.5f')
                    result = new_temp2
                caculated_pmi_pair.append(result)
            self.word_to_pmi_vec[word] = caculated_pmi_pair




    def computePmi(self,param1, param2):
        word = ''
        att = ''
        result = 0.0
        if param1 in self.feature_to_set_of_words and param2 in self.word_to_set_of_features:
            word = param2
            att = param1
        elif param2 in self.feature_to_set_of_words and param1 in self.word_to_set_of_features:
            word = param1
            att = param2
        else:
            return result
        if att in self.word_to_set_of_features[word]:
            pmi_vec = self.word_to_pmi_vec[word]
            index_dict = self.word_to_feature_to_index_dict[word]
            index = index_dict[att]
            result = pmi_vec[index]
        return result

    def cosine(self, u_common, v_common, u_full, v_full):
        u_common = np.array(u_common, dtype=float)
        v_common = np.array(v_common, dtype=float)
        u_full = np.array(u_full, dtype=float)
        v_full = np.array(v_full, dtype=float)
        dot_product = np.dot(u_common, v_common)
        norm_u = np.linalg.norm(u_full)
        norm_v = np.linalg.norm(v_full)

        return (dot_product / (norm_u*norm_v))

    def cosineDistance(self, target_word):
        dist = []
        words = []
        top_words = []
        #top_vals = []
        for word in self.word_to_pmi_vec:
            if word == target_word:
                continue
            returned_value = self.getPmiVecOrderedByCommonAttributes(target_word, word)
            cosine_val = self.cosine(returned_value[0], returned_value[1], returned_value[2], returned_value[3])
            dist.append(cosine_val)
            words.append(word)
        #get top 20
        for i in range(0, 20):
            max_val_index = dist.index(max(dist))
            top_words.append(words[max_val_index])
            words.pop(max_val_index)
            dist.pop(max_val_index)

        return top_words



    def getPmiVecOrderedByCommonAttributes(self, word1, word2):
        set_attributes_for_word1 = self.word_to_set_of_features[word1]
        set_attributes_for_word2 = self.word_to_set_of_features[word2]
        pmi_vec_word1 = []
        pmi_vec_word2 = []
        pmi_vec_word1_all_attributes = []
        pmi_vec_word2_all_attributes = []
        full_pmi_vec_word1 = self.word_to_pmi_vec[word1]
        full_pmi_vec_word2 = self.word_to_pmi_vec[word2]
        index_dict_word1 = self.word_to_feature_to_index_dict[word1]
        index_dict_word2 = self.word_to_feature_to_index_dict[word2]
        for feature_word1 in set_attributes_for_word1:
            pmi_vec_word1_all_attributes.append(full_pmi_vec_word1[index_dict_word1[feature_word1]])
            if feature_word1 not in set_attributes_for_word2:
                continue
            pmi_vec_word1.append(full_pmi_vec_word1[index_dict_word1[feature_word1]])
            pmi_vec_word2.append(full_pmi_vec_word2[index_dict_word2[feature_word1]])
        for feature_word2 in set_attributes_for_word2:
            pmi_vec_word2_all_attributes.append(full_pmi_vec_word2[index_dict_word2[feature_word2]])

        return [pmi_vec_word1, pmi_vec_word2, pmi_vec_word1_all_attributes, pmi_vec_word2_all_attributes]



if __name__ == '__main__':
    #
    target_words = ['car', 'bus', 'hospital', 'hotel', 'gun', 'bomb', 'horse', 'fox', 'table', 'bowl', 'guitar','piano']
    #create data class
    file_name = 'wikipedia.sample.trees.lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name, 100, 3)
    data_object.findCoOccurance(3)

    data_object.createPMIvectors()
    for target_word in target_words:
        words = data_object.cosineDistance(target_word)
        print("top words for target word " + target_word + " are: " + ', '.join(words))
    data_object.printHighestFeatures(target_words)
