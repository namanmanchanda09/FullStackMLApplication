from flask import Flask, request, jsonify
import numpy as np
import json
from tensorflow.keras.models import load_model
import joblib

from json import JSONEncoder


# class NumpyArrayEncoder(JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, numpy.ndarray):
#             return obj.tolist()
#         return JSONEncoder.default(self, obj)

def return_prediction(model,scaler,sample_json):
    
    
    s_len = sample_json["sepal_length"]
    s_wid = sample_json["sepal_width"]
    p_len = sample_json["petal_length"]
    p_wid = sample_json["petal_width"]
    
    flower=[[s_len,s_wid,p_len,p_wid]]
    classes = np.array(['setosa', 'versicolor', 'virginica'])
    flower=scaler.transform(flower)
    class_ind = model.predict_classes(flower)
    return classes[class_ind][0]

app=Flask(__name__)

@app.route("/")
def index():
    return '<h1>FLASK APP IS RUNNING</h1>'

flower_model = load_model('final_iris_model.h5')
flower_scaler = joblib.load('iris_scaler.pkl')

@app.route("/api/flower",methods=['POST'])
def flower_prediction():
    content = request.json
    
    results = return_prediction(flower_model,flower_scaler,content)
    return jsonify(results)


if __name__ == '__main__':
    app.run()






