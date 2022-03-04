import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import config_parameters as con


class TextData:
    def __init__(self, csvFile):
        self.trainData = pd.read_csv(csvFile)
        self.listColumnNames = list(self.trainData.columns)
        self.productColumnName = self.listColumnNames[con.PRODUCT_INDEX]
        self.consumerMessageColumnName = self.listColumnNames[con.CONSUMER_MESSAGE_INDEX]
        self.productTypes = self.trainData[self.productColumnName].value_counts()
        self.tokenizer = None
        self.inputData = None
        self.outputLabels = None
        self.outputLabelsValues = None
        self.classWeights = None

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
