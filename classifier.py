from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding, Conv1D, MaxPooling1D, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras import Sequential, models, saving
from keras.callbacks import ModelCheckpoint
import numpy as np
import config_parameters as con


class TextClassifier:
    def __init__(self, shape):
        self.checkpoint = None
        self.history = None
        if shape != 0:
            self.model = Sequential()
            self.model.add(Embedding(con.MAX_NB_WORDS, con.EMBEDDING_DIM, input_length=shape))
            self.model.add(SpatialDropout1D(0.25))
            self.model.add(LSTM(100, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
            self.model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))
            self.model.add(Dense(1, activation='relu'))
            self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
            self.isPreTrained = False
        else:
            self.model = models.load_model('output/regression_propuesta.h5')
            self.isPreTrained = True

    def showSummary(self):
        print(self.model.summary())

    def activateSaveCheckpoints(self):
        checkpoints_filepath = "tmp/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        self.checkpoint = ModelCheckpoint(checkpoints_filepath, monitor='val_loss', verbose=1,
                                          save_best_only=True, mode='min')

    def trainClassifier(self, inputData, outputLabels, epochs, batchSize):
        if self.isPreTrained is False:
            if self.checkpoint is None:
                self.history = self.model.fit(inputData, outputLabels, epochs=epochs, batch_size=batchSize,
                                              validation_split=0.15,
                                              callbacks=[EarlyStopping(monitor='val_loss', patience=3,
                                                                       min_delta=0.0001)])
            else:
                self.history = self.model.fit(inputData, outputLabels, epochs=epochs, batch_size=batchSize,
                                              validation_split=0.15,
                                              callbacks=[self.checkpoint,
                                                         EarlyStopping(monitor='val_loss', patience=3,
                                                                       min_delta=0.0001)])

    def classifyData(self, inputData, batchSize):
        scoresArray = self.model.predict(inputData, verbose=1, batch_size=batchSize)
        predictedClass = np.argmax(scoresArray, axis=1)
        predictedScore = np.max(scoresArray, axis=1)
        return predictedClass, predictedScore, scoresArray

    def saveModel(self, path):
        self.model.save(path)


class NLPClassifier:
    def __init__(self):
        self.model = saving.load_model('output/my_model.h5')

    def classifyData(self, inputData):
        return self.model.predict(inputData, verbose=1)
