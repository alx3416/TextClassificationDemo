import preprocessing as pre
import classifier as cla

model = cla.NLPClassifier()
messageData = pre.MessageData("mi_mensaje")
messageData.tokenizeData()
while messageData.getProtobuffStatus():
    if messageData.waitForMessage(50):
        print(messageData.getComment())
        print("Predicting comment rating...")
        messageData.paddingData()
        x = model.classifyData(messageData.inputData)
        print(x*5)
