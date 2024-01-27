import preprocessing as pre
import reporting as rep
import classifier as cla
import config_parameters as con


messageData = pre.MessageData("mi_mensaje")
while messageData.getProtobuffStatus():
    if messageData.waitForMessage(50):
        print(messageData.getComment())
