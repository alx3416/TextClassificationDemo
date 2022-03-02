import numpy as np
from keras.datasets import imdb
from keras.layers import LSTM, Embedding, Dense
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top 6000 words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=6000)
# pad input sequences
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)
#model
model = Sequential()
model.add(Embedding(6000, 32, input_length=500))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))