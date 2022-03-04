import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def barPlotClassesAndSave(productTypes):
    plt.bar(productTypes.index, productTypes.values)
    plt.xticks(rotation='vertical')
    plt.title('Dataset classes')
    plt.tight_layout()
    plt.savefig('plots/dataset_classes.png')
    plt.show()


def plotAccuracyHistory(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/AccuracyTrain.png')
    plt.show()


def plotLossHistory(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/LossTrain.png')
    plt.show()


def showConfusionMatrix(y_test, y_pred):
    print('Confusion Matrix')
    print(confusion_matrix(y_test, y_pred))


def showClassificationReport(y_test, y_pred, target_names):
    report = classification_report(y_test, y_pred, target_names=target_names)
    print(report)


def saveClassificationReport(y_test, y_pred, target_names):
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(report)
    pd.DataFrame(report).to_csv('output/classification_report.csv')


def saveHistory(history):
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'output/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
