import preprocessing as pre
import classifier as cla

model = cla.NLPClassifier()
messageData = pre.MessageData("mi_mensaje")
messageData.tokenizeData()
print("Service ready, waiting message")
while messageData.getProtobuffStatus():
    if messageData.waitForMessage(50):
        print(messageData.getComment())
        print("Predicting comment rating...")
        messageData.paddingData()
        x = model.classifyData(messageData.inputData)[0]
        print(str(round((x[0] * 4) + 1)) + " stars predicted")
