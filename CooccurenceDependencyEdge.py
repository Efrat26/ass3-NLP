import sys
import numpy as np

class Data:
    def __init__(self, fileName):
        self.data = []
        self.file_name = fileName

    def readData(self):
        counter = 0
        input_f = open(self.file_name, 'r')
        x = input_f.read().splitlines()

if __name__ == '__main__':
    #create data class
    file_name = 'wikipedia.sample.trees.lemmatized'
    if len(sys.argv) > 0:
        file_name = sys.argv[1]
    data_object = Data(file_name)
    data_object.readData()
    print("finished reading data")



