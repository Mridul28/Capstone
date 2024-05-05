import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model_stacking.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    final_features = scaler.transform(final_features)
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    if output == 0:
        result = "Fit & Fine !!"
    elif output == 1:
        result = "Parkinson's Disease Detected !!"
    
    return render_template('index.html', prediction_text='!! Result: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
