import pandas as pd
import matplotlib.pyplot as plt

trainData = pd.read_csv("data/customer-issues-train.csv")
listColumnNames = list(trainData.columns)
print('List of column names : ', listColumnNames)
productIndex = 1
productColumnName = listColumnNames[productIndex]
productTypes = trainData[productColumnName].value_counts()


plt.bar(productTypes.index, productTypes.values)
plt.xticks(rotation='vertical')
plt.title('Dataset classes')
plt.tight_layout()
plt.savefig('plots/dataset_classes.png')
plt.show()