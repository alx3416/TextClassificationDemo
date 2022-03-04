import preprocessing as pre
import reporting as rep
import classifier as cla
import config_parameters as con


if __name__ == '__main__':
    # Read and preprocess data
    dataContainer = pre.TextData("data/customer-issues-train.csv")
    rep.barPlotClassesAndSave(dataContainer.productTypes)
    dataContainer.cleanData()
    dataContainer.tokenizeData()
    dataContainer.paddingData()
    outputLabels = dataContainer.getOutputLabels()
    outputLabelsValues = dataContainer.getLabelsArray()
    classWeights = dataContainer.getClassWeights()

    # Train and evaluate model
    # Define inputSize = 0 to load pretrained model
    # inputSize = dataContainer.inputData.shape[1]
    inputSize = 0

    NLPClassifier = cla.TextClassifier(inputSize)
    NLPClassifier.showSummary()
    NLPClassifier.activateSaveCheckpoints()
    NLPClassifier.trainClassifier(dataContainer.inputData, outputLabels, con.epochs,
                                  con.batchSize, classWeights)

    # Get quality metrics and scores
    classes, scores = NLPClassifier.classifyData(dataContainer.inputData, con.batchSize)
    rep.showConfusionMatrix(classes, outputLabelsValues)
    rep.showClassificationReport(classes, outputLabelsValues, list(dataContainer.productTypes.index))

    # Saving figures and data
    NLPClassifier.saveModel('output/my_model.h5')
    rep.saveClassificationReport(classes, outputLabelsValues, list(dataContainer.productTypes.index))
    if NLPClassifier.isPreTrained is False:
        rep.saveHistory(NLPClassifier.history)
        rep.plotAccuracyHistory(NLPClassifier.history)
        rep.plotLossHistory(NLPClassifier.history)

