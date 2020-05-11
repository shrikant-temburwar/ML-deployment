import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    name = ["age", "income", "student", "cr_rat"]
    # int_features = [int(x) for x in request.form.values()]
    
    feat = []
    for n in name:
        v = request.form.get(n)
        feat.append(v)
    #print(int_features)
    final_features = [np.array(feat)]
    prediction = model.predict(final_features)
    
    if prediction == 0:
        output = 'Yes'
    else:
        output = 'No'


    return render_template('index.html', prediction_text='Buy Computer: {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)