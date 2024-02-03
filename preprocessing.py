import sys
import pandas as pd
import importlib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import config_parameters as con
import ecal.core.core as ecal_core
from ecal.core.subscriber import ProtoSubscriber


class TextData:
    def __init__(self, csvFile, CONSUMER_MESSAGE_INDEX):
        self.trainData = pd.read_csv(csvFile)
        self.originalData = self.trainData
        self.listColumnNames = list(self.trainData.columns)
        self.productColumnName = self.listColumnNames[con.PRODUCT_INDEX]
        self.consumerMessageColumnName = self.listColumnNames[CONSUMER_MESSAGE_INDEX]
        self.languageMessageColumnName = self.listColumnNames[con.LANGUAGE_INDEX]
        self.productTypes = self.trainData[self.productColumnName].value_counts()
        self.tokenizer = None
        self.inputData = None
        self.outputLabels = None
        self.outputLabelsValues = None
        self.classWeights = None

    def filterLanguage(self):
        self.trainData = self.originalData[self.originalData[self.languageMessageColumnName] == con.LANGUAGE_SELECTED]

    def cleanData(self):
        self.trainData = self.trainData.reset_index(drop=True)
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        def cleanText(text):
            text = text.lower()  # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text)
            text = BAD_SYMBOLS_RE.sub('', text)
            text = text.replace('x', '')
            #    text = re.sub(r'\W+', '', text)
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
            return text

        self.trainData[self.consumerMessageColumnName] = self.trainData[self.consumerMessageColumnName].apply(cleanText)
        self.trainData[self.consumerMessageColumnName] = self.trainData[self.consumerMessageColumnName].str.replace('\d+', '')

    def tokenizeData(self):
        self.tokenizer = Tokenizer(num_words=con.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        self.tokenizer.fit_on_texts(self.trainData[self.consumerMessageColumnName].values)

    def paddingData(self):
        self.inputData = self.tokenizer.texts_to_sequences(self.trainData[self.consumerMessageColumnName].values)
        self.inputData = pad_sequences(self.inputData, maxlen=con.MAX_SEQUENCE_LENGTH)

    def getOutputLabels(self):
        self.outputLabels = pd.get_dummies(self.trainData[self.productColumnName]).values
        return self.outputLabels

    def getClassWeights(self):
        self.classWeights = np.sum(self.productTypes.values) / (len(self.productTypes) * self.productTypes.values)
        return self.classWeights

    def getLabelsArray(self):
        self.outputLabelsValues = np.zeros(self.outputLabels.shape[0])
        for row in range(self.outputLabels.shape[0]):
            self.outputLabelsValues[row] = np.argmax(self.outputLabels[row])
        return self.outputLabelsValues


class MessageData(TextData):
    def __init__(self, messageName):
        self.messageName = messageName
        self.processName = "Python Protobuf Subscriber"
        self.start()
        self.messageWasReceived = False
        self.messageWasActivated = False
        self.subscriber = None
        self.ret = int()
        self.message = self.startSubscriber(self.messageName)()
        self.tokenizer = None
        self.inputData = None

    def __del__(self):
        self.subscriber.c_subscriber.destroy()
        return

    @staticmethod
    def getProto(topicName):
        Proto = importlib.import_module("proto." + topicName + "_pb2")
        return eval("Proto." + topicName)

    def startSubscriber(self, topicName):
        ProtoPb = self.getProto(topicName)
        self.subscriber = ProtoSubscriber(topicName, ProtoPb)
        self.messageWasReceived = False
        self.messageWasActivated = False
        return ProtoPb

    def start(self):
        ecal_core.initialize(sys.argv, self.processName)
        ecal_core.set_process_state(1, 1, "")

    def receive(self, waitTime):
        self.ret, self.message, timeStamp = self.subscriber.receive(waitTime)
        if self.ret != 0:
            self.messageWasReceived = True

    def waitForMessage(self, waitTime):
        self.messageWasReceived = False
        self.receive(waitTime)
        return self.messageWasReceived

    @staticmethod
    def getProtobuffStatus():
        return ecal_core.ok()

    def getComment(self):
        return self.message.comment

    def getId(self):
        return self.message.id

    def getDate(self):
        return self.message.date

    def tokenizeData(self):
        self.tokenizer = Tokenizer(num_words=con.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        trainData = pd.read_csv('data/train.csv')
        listColumnNames = list(trainData.columns)
        consumerMessageColumnName = listColumnNames[con.CONSUMER_MESSAGE_INDEX_TRAIN]
        self.tokenizer.fit_on_texts(trainData[consumerMessageColumnName].values)

    def paddingData(self):
        self.inputData = self.tokenizer.texts_to_sequences([self.getComment()])
        self.inputData = pad_sequences(self.inputData, maxlen=con.MAX_SEQUENCE_LENGTH)

class jsonData(TextData):
    def __init__(self):
        self.tokenizer = None
        self.inputData = None
        self.rawComment = None

    def setComment(self, comment):
        self.rawComment = comment

    def tokenizeData(self):
        self.tokenizer = Tokenizer(num_words=con.MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        trainData = pd.read_csv('data/train.csv')
        listColumnNames = list(trainData.columns)
        consumerMessageColumnName = listColumnNames[con.CONSUMER_MESSAGE_INDEX_TRAIN]
        self.tokenizer.fit_on_texts(trainData[consumerMessageColumnName].values)

    def paddingData(self):
        self.inputData = self.tokenizer.texts_to_sequences([self.rawComment])
        self.inputData = pad_sequences(self.inputData, maxlen=con.MAX_SEQUENCE_LENGTH)
