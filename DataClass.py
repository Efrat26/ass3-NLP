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
    def __init__(self, fileName):
        self.temp = 0
        self.linesInFile = []
        self.num_of_words = defaultdict(int)#key is the word as lemma form (stem)
        self.file_name = fileName
        # tags of content words based on penn tree bank tagset, we mostly chose the tags of nouns, verbs and adjectives
        self.content_words_tags = set(['JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'VB',
                                      'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'])
        self.function_words_lemma_form = set(['be', 'have', 'do', '\'s', '[', ']', 'either', 'or', 'and'])
        #self.threshold = 1
        self.threshold = 100
        self.word_to_index_mapping = {}
        ##simple start
        ### for type 3 dist vectors:
        self.word_to_dist_vec_type3 = {}
        self.word_to_set_of_features = {}
        self.feature_to_word_type3 = {}

        #try to make it quicker
        self.word_to_feature_to_index_dict_type3 = {}
        ###for PMI values
        self.pmi_word_type3 = {}
        self.pmi_att_type3 = {}
        self.pmi_word_att_type3 = {}
        self.total_co_occ = 0
        # to use less memory PMI vectors include only the values and not a pair of <feature, value>. the order
        #of the features is the same as in the distributional vectors
        self.word_to_pmi_vec = {}

        ### for type 1 & 2 dist vectors:
        self.feature_to_word = {}

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
                if type == 3:
                    self.findCoOccuranceForSentence(sentence, type)
                elif type == 2:
                    self.findCoOccuranceForSentence2(sentence, type)
                elif type == 1:
                    self.findCoOccuranceForSentence1(sentence, type)
                sentence = []
                #print('num of sentence: ' + str(num_of_Sentence))
            else:
                sentence.append(line)
        print('finished')
        if type == 3:
            print('number of features is: ' + str(len(self.feature_to_word_type3)))



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
                feature_child = splitted_sentences[int(target_word_is_daughter_id)-1][2] +  ' ' + 'is_daughter'
                self.addFeatureType3(feature_child, target_word)
            #case 3.1: target word points to preposition
            elif daughter_stem in self.prepsitions or daughter_tag == 'IN':
                returned_value = self.findNounPrepositionPointsTo(target_word, splitted_sentences)
                if returned_value != None:
                    feature_child = returned_value[0] + ' ' + daughter_stem + '-' + returned_value[1]
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
                                feature_parent = returned_value[0] + ' ' + \
                                                 current_sentence[2] + '-' + returned_value[1]
                                add_feature = True
                        # if the word's tag is in the list of the words that we are interested in them:
                        elif self.isContentWord(current_sentence[3]):
                            #create features for parent
                            feature_parent = current_sentence[2] + ' ' + 'is_parent'
                            add_feature = True
                        if add_feature:
                            self.addFeatureType3(feature_parent, target_word)
                            add_feature = False


    def addFeatureType3(self, feature, sentence):
        word_lemma = sentence[2]
        #map feature to the words contained it
        if feature in self.feature_to_word_type3:
            set_of_words = self.feature_to_word_type3[feature]
            set_of_words.add(word_lemma)
            self.feature_to_word_type3[feature] = set_of_words
        else:
            set_of_words = set()
            set_of_words.add(word_lemma)
            self.feature_to_word_type3[feature] = set_of_words

        #map the word to the distribtional vectors

        if word_lemma in self.word_to_dist_vec_type3:#word lemma was seen before
            dist_vec = self.word_to_dist_vec_type3[word_lemma]
            set_of_features = self.word_to_set_of_features[word_lemma]
            feature_index_dict = self.word_to_feature_to_index_dict_type3[word_lemma]
            if feature in set_of_features:
                #retrive the index of the feature in the dist_vec
                word_index = feature_index_dict[feature]
                dist_vec[word_index][1] += 1
                self.word_to_dist_vec_type3[word_lemma] = dist_vec
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
                new_feature_counter_pair = [feature, 1]
                dist_vec.append(new_feature_counter_pair)
                self.word_to_dist_vec_type3[word_lemma] = dist_vec
                #add the index of the feature
                feature_index_dict[feature] = len(set_of_features) - 1
                self.word_to_feature_to_index_dict_type3[word_lemma] = feature_index_dict
                return
        #word not in dictionary - need to add it
        else:
            dist_vec = []
            new_feature_counter_pair = [feature, 1]
            dist_vec.append(new_feature_counter_pair)
            self.word_to_dist_vec_type3[word_lemma] = dist_vec
            set_of_features = set()
            set_of_features.add(feature)
            self.word_to_set_of_features[word_lemma] = set_of_features

            map_feature_to_index = {}
            map_feature_to_index[feature] = 0
            self.word_to_feature_to_index_dict_type3[word_lemma] = map_feature_to_index

            return

    def findCoOccuranceForSentence2(self, sentence, type):
        for idx1, w1 in enumerate(sentence):
            word1 = w1.rstrip().split('\t')
            index1 = self.word_to_index_mapping[word1[2]]
            for idx2, w2 in enumerate(sentence):
                if (idx1 == idx2): continue
                word2 = w2.rstrip().split('\t')
                index2 = self.word_to_index_mapping[word2[2]]
                if (index1 not in self.word_to_set_of_features):
                    self.word_to_set_of_features[index1] = defaultdict(int)
                if (index2 not in self.feature_to_word):
                    self.feature_to_word[index2] = {index1}
                else:
                    self.feature_to_word[index2].add(index1)
                self.word_to_set_of_features[index1][index2] += 1

    def findCoOccuranceForSentence1(self, sentence, type):
        sentenceIndex = []
        for w in (sentence):
            word = w.rstrip().split('\t')
            sentenceIndex.append(self.word_to_index_mapping[word[2]])
            for idx, currIndexWord in enumerate(sentenceIndex):
                window = self.getWordInWindow(sentenceIndex, idx)
                if (currIndexWord not in self.word_to_set_of_features):
                    self.word_to_set_of_features[currIndexWord] = defaultdict(int)
                for windowWord in window:
                    if (windowWord not in self.feature_to_word):
                        self.feature_to_word[windowWord] = {currIndexWord}
                    else:
                        self.feature_to_word[windowWord].add(currIndexWord)
                    self.word_to_set_of_features[currIndexWord][windowWord] += 1

    def getWordInWindow(self, arr, index):
        windowWord = []
        length = len(arr)
        for x in range(index - 2, index + 2):
            if (x > 0 and x < length):
                windowWord.append(arr[x])
        return windowWord

    def createPMIvectors(self):
        #calculate #(*,*)
        number_of_co_occ_observed_in_corpus = 0
        for key in self.word_to_dist_vec_type3:
            list_of_dist_vecs_for_word = self.word_to_dist_vec_type3[key]
            for i in range(0, len(list_of_dist_vecs_for_word)):
                number_of_co_occ_observed_in_corpus += list_of_dist_vecs_for_word[i][1]
        #number_of_co_occ_observed_in_corpus *= 2
        if number_of_co_occ_observed_in_corpus == 0:
            print("number of co-occurances is zero!")
            return
        else:
            print("total co-occ is " + str(number_of_co_occ_observed_in_corpus))
            self.total_co_occ = number_of_co_occ_observed_in_corpus

        for feature in self.feature_to_word_type3:
            list_of_words_has_feature = self.feature_to_word_type3[feature]
            # print(list_of_words_has_feature)
            #print("feature is: " + feature)
            #print("set of words with this feature: " + ' '.join(list_of_words_has_feature))
            counter = 0
            for word_with_feature in list_of_words_has_feature:
                #  print("word with feature is: " + word_with_feature)
                word_to_features_index_dict = self.word_to_feature_to_index_dict_type3[word_with_feature]
                index = word_to_features_index_dict[feature]
                # print("index is: " + str(index))
                dist_vec = self.word_to_dist_vec_type3[word_with_feature]
                counter += dist_vec[index][1]
            self.pmi_att_type3[feature] = counter / number_of_co_occ_observed_in_corpus

        # sanity check
        sanity_check_sum_of_features_probabilities = 0
        for feature in self.pmi_att_type3:
            sanity_check_sum_of_features_probabilities += self.pmi_att_type3[feature]
        print("sainty check: sum of probabilities for words " + str(sanity_check_sum_of_features_probabilities))

        # calculate p(word)
        for word in self.word_to_set_of_features:
            list_of_dist_vecs_for_word = self.word_to_dist_vec_type3[word]
            counter = 0
            for i in range(0,len(list_of_dist_vecs_for_word)):
                counter += list_of_dist_vecs_for_word[i][1]
            self.pmi_word_type3[word] = counter / number_of_co_occ_observed_in_corpus

        #sainty check
        sanity_check_sum_of_words_probabilities = 0
        for word in self.pmi_word_type3:
            sanity_check_sum_of_words_probabilities += self.pmi_word_type3[word]
        print("sanity check: sum of probabilities for words " + str(sanity_check_sum_of_words_probabilities))
        counter = 0
        #calculate p(word,att) & pmi vector (includes the sanity check counter)
        for word in self.word_to_dist_vec_type3:
            pmi_vec = []
            dist_vec = self.word_to_dist_vec_type3[word]
            for att in dist_vec:
                p_att_word = att[1] / number_of_co_occ_observed_in_corpus
                pair = [att[0], p_att_word]
                pmi_vec.append(pair)
                counter += pair[1]
            self.pmi_word_att_type3[word] = pmi_vec
        #sanity check
        print("sanity check for sigma on word,att result is: " + str(counter))

        #create PMI vector
        for word in self.pmi_word_att_type3:
            caculated_pmi_pair = []
            pmi_vec = self.pmi_word_att_type3[word]
            for i in range(0, len(pmi_vec)):
                current_pair = pmi_vec[i]
                p_att = self.pmi_att_type3[current_pair[0]]
                p_word = self.pmi_word_type3[word]
                p_att_word = current_pair[1]
                if p_att == 0 or p_word == 0:
                    result = 0.0
                elif p_att_word == 0:
                    result = 0.0
                else:
                    temp1 = p_att_word / (p_word * p_att)
                    temp2 = math.log(temp1)
                    #print(str(temp2))
                    new_temp2 = format(temp2, '.3f')
                    result = new_temp2
                caculated_pmi_pair.append(result)
            self.word_to_pmi_vec[word] = caculated_pmi_pair




    def computePmi(self,param1, param2):
        word = ''
        att = ''
        result = 0.0
        if param1 in self.feature_to_word_type3 and param2 in self.word_to_set_of_features:
            word = param2
            att = param1
        elif param2 in self.feature_to_word_type3 and param1 in self.word_to_set_of_features:
            word = param1
            att = param2
        else:
            return result
        if att in self.word_to_set_of_features[word]:
            pmi_vec = self.word_to_pmi_vec[word]
            index_dict = self.word_to_feature_to_index_dict_type3[word]
            index = index_dict[att]
            result = pmi_vec[index]
        return result

    def cosine(self, u_common, v_common, u_full, v_full):
        u_common= np.array(u_common, dtype=float)
        v_common = np.array(v_common, dtype=float)
        dot_product = np.dot(u_common, v_common)
        u_full = np.array(u_full, dtype=float)
        v_full = np.array(v_full, dtype=float)
        norm_u = np.linalg.norm(u_full)
        norm_v = np.linalg.norm(v_full)
        return dot_product / (norm_u * norm_v)

    def cosineDistance(self, target_word):
        dist = []
        words = []
        top_words = []
        top_vals = []
        for word in self.word_to_pmi_vec:
            if word == target_word:
                continue
            returned_value = self.getPmiVecOrderedByCommonAttributes(target_word, word)
            cosine_val = self.cosine(returned_value[0], returned_value[1], returned_value[2], returned_value[3])
            dist.append(cosine_val)
            words.append(word)
        #get top 20
        min_val_index = 0
        min_val_value = np.NINF
        for i in range(0, len(dist)):
            if top_vals.__len__() < 20:
                top_vals.append(dist[0])
                top_words.append(words[0])
                if dist[0] < min_val_value:
                    min_val_value = dist[0]
                    min_val_index = top_vals.__len__() - 1
                dist.pop(0)
                words.pop(0)
            else:
                if dist[0] > min_val_value:
                    top_vals.pop(min_val_index)
                    top_words.pop(min_val_index)
                    top_vals.append(dist[0])
                    top_words.append(words[0])
                    min_val_index = top_vals.index(min(top_vals))
                    min_val_value = top_vals[min_val_index]
                dist.pop(0)
                words.pop(0)

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
        index_dict_word1 = self.word_to_feature_to_index_dict_type3[word1]
        index_dict_word2 = self.word_to_feature_to_index_dict_type3[word2]
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
    file_name = 'wikipedia_sample_trees_lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)
    data_object.findCoOccurance(3)

    data_object.createPMIvectors()
    words = data_object.cosineDistance(target_words[0])
    print("top words for target word: " + target_words[0] + " are: " + ','.join(words))
