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

    # Get quality metrics and scores
    classes, scores, scoresFull = NLPClassifier.classifyData(dataContainer.inputData, con.batchSize)
    rep.showConfusionMatrix(classes, outputLabelsValues)
    rep.showClassificationReport(classes, outputLabelsValues, list(map(str, (dataContainer.productTypes.index))))

    # Saving figures and data
    NLPClassifier.saveModel('output/my_model.h5')
    rep.saveClassificationReport(classes, outputLabelsValues, list(dataContainer.productTypes.index))
    if NLPClassifier.isPreTrained is False:
        rep.saveHistory(NLPClassifier.history)
        rep.plotAccuracyHistory(NLPClassifier.history)
        rep.plotLossHistory(NLPClassifier.history)

    dataTest = pre.TextData("data/test.csv", con.CONSUMER_MESSAGE_INDEX_TEST)
    dataTest.filterLanguage()
    dataTest.cleanData()
    dataTest.tokenizeData()
    dataTest.paddingData()
    classesTest, scoresTest, scoresFullTest = NLPClassifier.classifyData(dataTest.inputData, con.batchSize)
    # rep.generatePredictionTestCSV(dataTest, classesTest, dataContainer.productTypes.index)
    # rep.generateScoreTestCSV(dataTest, scoresFullTest)





