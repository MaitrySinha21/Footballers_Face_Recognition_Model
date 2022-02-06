from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from com_in_ineuron_ai_utils.utils import decodeImage
from predict import Classifier

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
CORS(application)




#@cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = Classifier(self.filename)



@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')
    


@application.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.prediction()
    return jsonify(result)

clApp = ClientApp()
port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = ClientApp()
    #application.run(host='0.0.0.0', port=port)
    application.run(host='0.0.0.0', port=8000, debug=True)
