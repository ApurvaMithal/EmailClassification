# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:25:50 2018

@author: Apurva
"""

import os
import sys
import collections
import re
import math
import copy

training_set = dict()
test_set = dict()

trainSetWithoutStopWord = dict()
testSetWithoutStopWord = dict()

stop_words = []

classes = ["ham", "spam"]

conditional_prob = dict()
conditional_prob_withoutStopWords = dict()
prior = dict()
prior_WithoutStopWords = dict()

def getData(dataSet, directory, target):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r') as text_file:
                # stores dictionary of dictionary of dictionary as explained above in the initialization
                text = text_file.read()
                dataSet.update({dir_entry_path: Mail(text, bagOfWords(text), target)})


def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)


def getVocab(data_set):
    all_text = ""
    vocab = []
    for x in data_set:
        all_text += data_set[x].getText()
    for y in bagOfWords(all_text):
        vocab.append(y)
    return vocab

def setStopWords():
    stopWords = []
    with open('stop_words.txt', 'r') as txt:
        stopWords = (txt.read().splitlines())
    return stopWords


def deleteStopWords(stops, data_set):
    dataSetWithoutStopWords = copy.deepcopy(data_set)
    for i in stops:
        for j in dataSetWithoutStopWords:
            if i in dataSetWithoutStopWords[j].getWordFreqs():
                del dataSetWithoutStopWords[j].getWordFreqs()[i]
    return dataSetWithoutStopWords


def multinomialNaiveBayes(training, priors, cond):
    v = getVocab(training)
    n = len(training)
    for c in classes:
        n_c = 0.0
        text_string = ""
        for i in training:
            if training[i].getTarget() == c:
                n_c += 1
                text_string += training[i].getText()
        priors[c] = float(n_c) / float(n)
        token_freqs = bagOfWords(text_string)
        for t in v:
            if t in token_freqs:
                cond.update({t + "_" + c: (float((token_freqs[t] + 1.0)) / float((len(text_string) + len(token_freqs))))})
            else:
                cond.update({t + "_" + c: (float(1.0) / float((len(text_string) + len(token_freqs))))})

def multinomialNaiveBayesCalculation(data_instance, priors, cond):
    score = {}
    for c in classes:
        score[c] = math.log10(float(priors[c]))
        for t in data_instance.getWordFreqs():
            if (t + "_" + c) in cond:
                score[c] += float(math.log10(cond[t + "_" + c]))
    if score["spam"] > score["ham"]:
        return "spam"
    else:
        return "ham"

		
class Mail:
    text = ""
    word_freqs = {}

    target = ""
    learned_class = ""

    # Constructor
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


def main():

    training_spam_dir = "./train/spam/"
    training_ham_dir = "./train/ham/"
    test_spam_dir = "./test/spam/"
    test_ham_dir = "./test/ham/"
	
    getData(training_set, training_spam_dir, classes[1])
    getData(training_set, training_ham_dir, classes[0])
    getData(test_set, test_spam_dir, classes[1])
    getData(test_set, test_ham_dir, classes[0])

    stop_words = setStopWords()

    trainSetWithoutStopWord = deleteStopWords(stop_words, training_set)
    testSetWithoutStopWord = deleteStopWords(stop_words, test_set)

    multinomialNaiveBayes(training_set, prior, conditional_prob)
    multinomialNaiveBayes(trainSetWithoutStopWord, prior_WithoutStopWords, conditional_prob_withoutStopWords)

    numberOfCorrectClassification = 0
    for i in test_set:
        test_set[i].setLearnedClass(multinomialNaiveBayesCalculation(test_set[i], prior, conditional_prob))
        if test_set[i].getLearnedClass() == test_set[i].getTarget():
            numberOfCorrectClassification += 1

    numberOfCorrectClassification_withoutStopWords = 0
    for i in testSetWithoutStopWord:
        testSetWithoutStopWord[i].setLearnedClass(multinomialNaiveBayesCalculation(testSetWithoutStopWord[i], prior_WithoutStopWords,
                                                                conditional_prob_withoutStopWords))
        if testSetWithoutStopWord[i].getLearnedClass() == testSetWithoutStopWord[i].getTarget():
            numberOfCorrectClassification_withoutStopWords += 1

    print ("Naive Bayes: Number of Correct guesses before removing stop words:\t%d/%s" % (numberOfCorrectClassification, len(test_set)))
    print ("Naive Bayes: Accuracy before removing stop words:\t\t\t%.4f%%" % (100.0 * float(numberOfCorrectClassification) / float(len(test_set))))
    print ("Naive Bayes: Number of Correct guesses after removing stop words:\t\t%d/%s" % (numberOfCorrectClassification_withoutStopWords, len(testSetWithoutStopWord)))
    print ("Naive Bayes: Accuracy after removing stop words:\t\t\t%.4f%%" % (100.0 * float(numberOfCorrectClassification_withoutStopWords) / float(len(testSetWithoutStopWord))))

if __name__ == '__main__':
    main()