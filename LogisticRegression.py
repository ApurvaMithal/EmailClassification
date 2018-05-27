# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 02:04:19 2018

@author: Apurva
"""

import os
import collections
import sys
import re
import copy
import math

classes = ["ham", "spam"]
stop_words = []

training_set = dict()
trainSetWithoutStopWord = dict()
training_set_vocab = []
trainSetWithoutStopWord_vocab = []

test_set = dict()
testSetWithoutStopWord = dict()

weights = {'weight_zero': 0.0}
weightsWithoutStopWord = {'weight_zero': 0.0}

learningRate = .001
regularizationConst = 0.0

def getData(dataSet, directory, target):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                text = text_file.read()
                dataSet.update({dir_entry_path: Mail(text, bagOfWords(text), target)})

def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)

def setStopWords():
    stopWords = []
    with open('stop_words.txt', 'r') as txt:
        stopWords = (txt.read().splitlines())
    return stopWords


def deleteStopWords(stopWords, data_set):
    dataSetWithoutStopWords = copy.deepcopy(data_set)
    for i in stopWords:
        for j in dataSetWithoutStopWords:
            if i in dataSetWithoutStopWords[j].getWordFreqs():
                del dataSetWithoutStopWords[j].getWordFreqs()[i]
    return dataSetWithoutStopWords
	
				
def getVocabulary(data_set):
    vocab = []
    for i in data_set:
        for j in data_set[i].getWordFreqs():
            if j not in vocab:
                vocab.append(j)
    return vocab


def learnWeights(trainingSet, weights_param, iter, reg):
    for x in range(0, iter):
        print("iteration ", x)
        counter = 1
        for w in weights_param:
            sum = 0.0
            for i in trainingSet:
                y = 0.0
                if trainingSet[i].getTarget() == classes[1]:
                    y = 1.0
                if w in trainingSet[i].getWordFreqs():
                    sum += float(trainingSet[i].getWordFreqs()[w]) * (y - sigmoidFunc(classes[1], weights_param, trainingSet[i]))
            weights_param[w] += ((learningRate * sum) - (learningRate * float(reg) * weights_param[w]))


def sigmoidFunc(target, weights_param, doc):
    if target == classes[0]:
        sum_wx_0 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            sum_wx_0 += weights_param[i] * float(doc.getWordFreqs()[i])
        return 1.0 / (1.0 + math.exp(float(sum_wx_0)))

    elif target == classes[1]:
        sum_wx_1 = weights_param['weight_zero']
        for i in doc.getWordFreqs():
            if i not in weights_param:
                weights_param[i] = 0.0
            sum_wx_1 += weights_param[i] * float(doc.getWordFreqs()[i])
        return math.exp(float(sum_wx_1)) / (1.0 + math.exp(float(sum_wx_1)))


def logisticCalculation(data_point, weights_param):
    score = {}
    score[0] = sigmoidFunc(classes[0], weights_param, data_point)
    score[1] = sigmoidFunc(classes[1], weights_param, data_point)
    if score[1] > score[0]:
        return classes[1]
    else:
        return classes[0]


class Mail:
    text = ""
    word_freqs = {'weight_zero': 1.0}

    target = ""
    learned_class = ""

    def __init__(self, text, counter, target):
        self.text = text
        self.word_freqs = counter
        self.target = target

    def getText(self):
        return self.text

    def getWordFreqs(self):
        return self.word_freqs

    def getTarget(self):
        return self.target

    def getLearnedClass(self):
        return self.learned_class

    def setLearnedClass(self, guess):
        self.learned_class = guess


def main(lambdaConstant, numIter):
    if len(sys.argv) < 3:
        print("There should be 2 arguments -- lambda noOfIterations")
        sys.exit(1)
		
    num_iterations = int(numIter)
    training_spam_dir = "./train/spam/"
    training_ham_dir = "./train/ham/"
    test_spam_dir = "./test/spam/"
    test_ham_dir = "./test/ham/"
	
    getData(training_set, training_spam_dir, classes[1])
    getData(training_set, training_ham_dir, classes[0])
    getData(test_set, test_spam_dir, classes[1])
    getData(test_set, test_ham_dir, classes[0])
    regularizationConst = lambdaConstant

    stop_words = setStopWords()

    trainSetWithoutStopWord = deleteStopWords(stop_words, training_set)
    testSetWithoutStopWord = deleteStopWords(stop_words, test_set)

    training_set_vocab = getVocabulary(training_set)
    trainSetWithoutStopWord_vocab = getVocabulary(trainSetWithoutStopWord)

    for i in training_set_vocab:
        weights[i] = 0.0
    for i in trainSetWithoutStopWord_vocab:
        weightsWithoutStopWord[i] = 0.0

    learnWeights(training_set, weights, num_iterations, regularizationConst)
    learnWeights(trainSetWithoutStopWord, weightsWithoutStopWord, num_iterations, regularizationConst)


    numberOfCorrectClassification = 0.0
    for i in test_set:
        test_set[i].setLearnedClass(logisticCalculation(test_set[i], weights))
        if test_set[i].getLearnedClass() == test_set[i].getTarget():
            numberOfCorrectClassification += 1.0

    numberOfCorrectClassification_withoutStopWords = 0.0
    for i in testSetWithoutStopWord:
        testSetWithoutStopWord[i].setLearnedClass(logisticCalculation(testSetWithoutStopWord[i], weightsWithoutStopWord))
        if testSetWithoutStopWord[i].getLearnedClass() == testSetWithoutStopWord[i].getTarget():
            numberOfCorrectClassification_withoutStopWords += 1.0

    print ("Logistic Regression: Number of Correct Guesses before removing the stop words:\t%d/%s" % (numberOfCorrectClassification, len(test_set)))
    print ("Logistic Regression: Accuracy before removing stop words:\t\t\t%.4f%%" % (100.0 * float(numberOfCorrectClassification) / float(len(test_set))))
    
    print ("Logistic Regression: Number of Correct guesses after removing stop the words:\t\t%d/%s" % (numberOfCorrectClassification_withoutStopWords, len(testSetWithoutStopWord)))
    print ("Logistic Regression: Accuracy after removing stop words:\t\t\t%.4f%%" % (100.0 * float(numberOfCorrectClassification_withoutStopWords) / float(len(testSetWithoutStopWord))))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])