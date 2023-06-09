import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
 return render_template('park.html')
@app.route('/predict',methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    if output == 0:
        return render_template('park.html', prediction_text='THE PATIENT DOESNOT HAVE A PARKINSONS DISEASE')
    else:
        return render_template('park.html', prediction_text='THE PATIENT HAVE A PARKINSONS DISEASE')
if __name__ == "__main__":
    app.run(debug=False)