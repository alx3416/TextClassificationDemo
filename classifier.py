from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding, Bidirectional
from keras.callbacks import EarlyStopping
from keras import Sequential, Model, Input
from keras.callbacks import ModelCheckpoint
import numpy as np
import config_parameters as con


class TextClassifier:
    def __init__(self, shape):
        self.checkpoint = None
        self.history = None
        self.model = Sequential()
        self.model.add(Embedding(con.MAX_NB_WORDS, con.EMBEDDING_DIM, input_length=shape))
        self.model.add(SpatialDropout1D(0.25))
        self.model.add(LSTM(100, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
        self.model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(12, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def showSummary(self):
        print(self.model.summary())

    def activateSaveCheckpoints(self):
        checkpoints_filepath = "tmp/weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
        self.checkpoint = ModelCheckpoint(checkpoints_filepath, monitor='val_accuracy', verbose=1,
                                          save_best_only=True, mode='max')

    def trainClassifier(self, inputData, outputLabels, epochs, batchSize, class_weights):
        if self.checkpoint is None:
            self.history = self.model.fit(inputData, outputLabels, epochs=epochs, batch_size=batchSize,
                                          validation_split=0.15,
                                          class_weight=class_weights,
                                          callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
        else:
            self.history = self.model.fit(inputData, outputLabels, epochs=epochs, batch_size=batchSize,
                                          validation_split=0.15,
                                          class_weight=class_weights,
                                          callbacks=[self.checkpoint,
                                                     EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

    def classifyData(self, inputData, batchSize):
        scoresArray = self.model.predict(inputData, verbose=1, batch_size=batchSize)
        predictedClass = np.argmax(scoresArray, axis=1)
        predictedScore = np.max(scoresArray, axis=1)
        return predictedClass, predictedScore

    def saveModel(self, path):
        self.model.save(path)
