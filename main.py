import preprocessing as pre
import reporting as rep
import classifier as cla
import config_parameters as con


if __name__ == '__main__':
    # Read and preprocess data
    dataContainer = pre.TextData("data/train.csv", con.CONSUMER_MESSAGE_INDEX_TRAIN)
    dataContainer.filterLanguage()
    rep.barPlotClassesAndSave(dataContainer.productTypes)
    dataContainer.cleanData()
    dataContainer.tokenizeData()
    dataContainer.paddingData()
    outputLabels = dataContainer.getOutputLabels()
    outputLabelsValues = dataContainer.getLabelsArray()

    # Train and evaluate model
    # Define inputSize = 0 to load pretrained model
    inputSize = dataContainer.inputData.shape[1]
    # inputSize = 0

    NLPClassifier = cla.TextClassifier(inputSize)
    NLPClassifier.showSummary()
    NLPClassifier.activateSaveCheckpoints()
    NLPClassifier.trainClassifier(dataContainer.inputData, outputLabelsValues/4.0, con.epochs,
                                  con.batchSize)



    # Saving figures and data
    NLPClassifier.saveModel('output/my_model_regression.h5')






