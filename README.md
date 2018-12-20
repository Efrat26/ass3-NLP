# NLP - assignment 3

Distributional Semantics

### Prerequisites

we used python 3.6 interpreter


## Running the script



### Distributional Vectors & PMI

You should run the DataClass.py

in the main - when you create the DataClass, you need to specify the threshold for the content words and the
type of distributional vectors you would like.

DataClass.findCoOccurance() - finds the co-occurances for the type specified

DataClass..createPMIvectors() - creates the PMI vectors

DataClass..cosineDistance(target_word) - finds the most similar words

DataClass.printHighestFeatures(target_words_list) - prints the highest features

### Word2Vec

You should run the ass3.py

The script get 2 parameters:
* bow5.words / deps.words 
* bow5.contexts / deps.contexts
respectively.

in the main - you create w2v class and run the function:
w2v.load_words_and_vectors(file_name_words) - read the word file

w2v.create_word_to_vec() - create vectors

w2v.find_sim_word() - find similar vector words to the target words

w2v.load_words_and_vectors(file_name_contexts) - read the contexts file

w2v.printResults() - print the results of the similar words and features.



## Authors

* **Efrat Soffer** & **Osnat Drien**

