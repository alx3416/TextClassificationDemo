import json
import preprocessing as pre
import classifier as cla
from flask import request

from flask import Flask, render_template

model = cla.NLPClassifier()
messageData = pre.jsonData()
messageData.tokenizeData()
print("Service ready, waiting message")

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    print("The received comment is: " + output)
    messageData.setComment(output)
    print("Predicting comment rating...")
    messageData.paddingData()
    modelResult = model.classifyData(messageData.inputData)[0]
    result = str(round((modelResult[0] * 4) + 1)) + " stars predicted"
    print(result)
    return result
