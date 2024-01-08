from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route("/")
def helloWorld():
    input_data = [8.064170137236562,209.83933337513497,8712.00258845446,8.591510126308991,355.44372109287366,338.8481579462765,18.72131156504708,106.2430659759468,3.246077223248075]
    input_data_reshaped = np.array(input_data).reshape(1, -1)
    scaled_features = scaler.transform(input_data_reshaped)
    prediction = model.predict(scaled_features)[0]

    return jsonify({"prediction": int(prediction)})
    

@app.route('/predict', methods=['POST', 'GET', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":  
        response = jsonify({'message': 'CORS preflight response'})
    elif request.method == "POST":
        try:
            keys = ['ph', 'hardness', 'solids', 'chloramines', 'sulfate', 'conductivity', 'organic_carbon', 'trihalomethanes', 'turbidity']
            features = [float(request.json[key]) for key in keys]
            scaled_features = scaler.transform(np.array(features).reshape(1, -1))
            prediction = model.predict(scaled_features)
            print("Prediction result:", int(prediction[0])) 
            response = jsonify({"prediction": int(prediction[0])})
        except Exception as e:
            response = jsonify({"error": str(e)})
    elif request.method == "GET":
        response=jsonify({"prediction": ""})
    else:
        raise RuntimeError("Weird - don't know how to handle method {}".format(request.method))
    
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    
    return response

if __name__ == '__main__':
    app.run(port=8000)
   
#activate venv: venv\Scripts\activate