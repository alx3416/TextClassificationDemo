import preprocessing as pre
import reporting as rep
import classifier as cla
import config_parameters as con


def main():
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
    NLPClassifier = cla.TextClassifier(dataContainer.inputData.shape[1])
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
    rep.saveHistory(NLPClassifier.history)