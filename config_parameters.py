
# CSV related params
CSV_DATA_TRAIN_PATH = "data/train.csv"
CSV_DATA_TEST_PATH = "data/test.csv"
PRODUCT_INDEX = 4  # output
LANGUAGE_INDEX = 7
LANGUAGE_SELECTED = "en"
#  en -> english
#  de -> deutchsland
#  es -> español
#  fr -> french
#  ja -> japanese
#  zh -> chinese

CONSUMER_MESSAGE_INDEX_TRAIN = 5  # input
CONSUMER_MESSAGE_INDEX_TEST = 5  # input

# Word2Vec params
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

# NLP model train params
epochs = 40
batchSize = 512