from numpy import *

def loadDataSet():
    # training data
    postingList = [['my', 'dog', 'has', 'flea', \
                         'problems', 'help', 'please'],
                         ['maybe', 'not', 'take', 'him', \
                          'to', 'dog', 'park', 'stupid'],
                         ['my', 'dalmation', 'is', 'so', 'cute', \
                           'I', 'love', 'him'],
                         ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                         ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
                           'to', 'stop', 'him'],
                         ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # training labels
    classVec = [0, 1, 0, 1, 0, 1]
    return postingList, classVec

# create my vocabulary list from a dataSet
# the return value is a vector of words
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

# Words to vector: 
# if a word emerges in the inputSet and also my vocabulary list, then mark it as 1 in the feature vector
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word {} is not in my Vocabulary!".format(word))
    return returnVec

# bag of words model
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            # key difference to setOfWords2Vec: increment the counter instead of setting it to 1
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word {} is not in my Vocabulary!".format(word))
    return returnVec


# train the words and generate probability vectors
# p(c1), p(w|c1), p(w|c0)
def trainNB0(trainMatrix, trainCategory):
    # sample count
    numTrainDocs = len(trainMatrix)
    # feature count(word count/column count in each sample)
    numWords = len(trainMatrix[0]) 
    # the probability of being abusive, which is p(c1)
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    # use ones instead of zeros to get rid of probability 0 when multiuplying all p(w|c)
    # since we assume that all wi|c are independent
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # distinguish class 1 and class 0 with two set of variables
        if trainCategory[i] == 1:
            # the trainMatrix[i] is the vector of words of training sample i
            p1Num += trainMatrix[i]
            # the sum represents how many 1(abusive) occurs
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # p(w|c1)
    p1Vect = log(p1Num / p1Denom)
    # p(w|c0)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

# Naive Bayes classfication
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # log(p) will change multiplications into sum
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    print("P0 = {}; P1 = {}".format(p0, p1))
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB(testEntries):
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    # prepare training data
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # train
    p0V, p1V, pAb = trainNB0(array(trainMat), array(listClasses))
    for testEntry in testEntries:
        # word2vec for testEntry
        thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
        # classify testEntry!
        print("{} classified as: {}".format(testEntry, classifyNB(thisDoc, p0V, p1V, pAb)))
