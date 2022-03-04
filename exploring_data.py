import pandas as pd
import matplotlib.pyplot as plt
# for preprocessing
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.utils import class_weight
import numpy as np
from keras.utils import to_categorical
# for LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import SpatialDropout1D, LSTM, Dense, Embedding, Bidirectional
from keras.callbacks import EarlyStopping
from keras import Sequential, Model, Input
from keras.callbacks import ModelCheckpoint
# for reporting
from sklearn.metrics import classification_report, confusion_matrix

trainData = pd.read_csv("data/customer-issues-train.csv")
listColumnNames = list(trainData.columns)
print('List of column names : ', listColumnNames)
productIndex = 1
consumerMessageIndex = 5
productColumnName = listColumnNames[productIndex]
consumerMessageColumnName = listColumnNames[consumerMessageIndex]
productTypes = trainData[productColumnName].value_counts()

for row in range(9):
    print(trainData.loc[row].at[consumerMessageColumnName])

plt.bar(productTypes.index, productTypes.values)
plt.xticks(rotation='vertical')
plt.title('Dataset classes')
plt.tight_layout()
plt.savefig('plots/dataset_classes.png')
plt.show()

# preprocessing consumer mesages
df = trainData.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    """
        text: a string

        return: modified initial string
    """
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    #    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text


trainData[consumerMessageColumnName] = trainData[consumerMessageColumnName].apply(clean_text)
trainData[consumerMessageColumnName] = trainData[consumerMessageColumnName].str.replace('\d+', '')

for row in range(9):
    print(trainData.loc[row].at[consumerMessageColumnName])

# LSTM Modelling
# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(trainData[consumerMessageColumnName].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# truncate and pad all same length for modelling
inputData = tokenizer.texts_to_sequences(trainData[consumerMessageColumnName].values)
inputData = pad_sequences(inputData, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', inputData.shape)

# getting numbers as outputs for trainning
outputLabels = pd.get_dummies(trainData[productColumnName]).values
print('Shape of label tensor:', outputLabels.shape)


# class weights balancing dataset
def getLabelsArray(outputLabels):
    labelsArray = np.zeros(outputLabels.shape[0])
    for row in range(outputLabels.shape[0]):
        labelsArray[row] = np.argmax(outputLabels[row])
    return labelsArray


outputArray = getLabelsArray(outputLabels)
# categorical KERAS
categorical_labels = to_categorical(outputArray, num_classes=12)
# class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(outputArray), y=outputArray)
class_weights = np.sum(productTypes.values)/(12*productTypes.values)

# Embed each integer in a 128-dimensional vector
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=inputData.shape[1]))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(100, dropout=0.25, recurrent_dropout=0.25, return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(12, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

filepath = "tmp/weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

epochs = 10
batch_size = 1024
history = model.fit(inputData, outputLabels, epochs=epochs, batch_size=batch_size, validation_split=0.15,
                    class_weight=class_weights,
                    callbacks=[checkpoint, EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Reporting and quality metrics
#score = model.evaluate(inputData, outputLabels, verbose=1)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

y_pred = model.predict(inputData, verbose=1, batch_size=batch_size)
y_pred = np.argmax(y_pred, axis=1)
y_test = outputArray
target_names = list(productTypes.index)
print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
#target_names = ['0', '1', '2','3','4','5','6','7','8','9']
print(classification_report(y_test, y_pred, target_names=target_names))

# saving
model.save('output/RNN_Bidirectional_balancedManualfull.h5')
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'output/history_balancedManualfull.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

