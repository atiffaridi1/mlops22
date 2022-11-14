# Using flask to make an api
# import necessary libraries and functions
from flask import Flask, jsonify, request
import pickle
# creating a Flask app
app = Flask(__name__)


model_path = "model.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods = ['GET', 'POST'])
def home():
        if(request.method == 'GET'):

                data = "hello world"
                return jsonify({'data': data})




@app.route('/predict', methods=['POST'])
def predict():
    image1 = request.json['image1']
    image2 = request.json['image2']
    print("done loading")
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    if predicted1==predicted2:
        return {"matched":True}
    else:
        return {"matched":False}
