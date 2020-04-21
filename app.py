import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

features = []

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features.clear()
    if(request.form['submit_button']=='High'):
           features.append(1)
    if(request.form['submit_button']=='Yes'):
           features.append(1)      
    message3 = features
    #int_features = [int(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)

    output = 9
    if(output==1):
        message1='These symptoms are of Cold. Please do not panic, below are the hospitals which are suggected for the same.'
        message2 = '1. Jupiter hospital , Balewadi        2. Life Point, near Sayaji hotel, Wakad'
    if(output==2):
        message1='These symptoms looks similar to Covid. Please do not panic, below are the hospitals which are suggected for the same.'
        message2 = '1. Jupiter hospital , Balewadi        2. Life Point, near Sayaji hotel, Wakad'
    if(output==3):
        message1='These symptoms are of Fever. Please do not panic, below are the hospitals which are suggected for the same.'
        message2 = '1. Jupiter hospital , Balewadi        2. Life Point, near Sayaji hotel, Wakad'
    if(output==4):
        message1='These symptoms are of Flu. Please do not panic, below are the hospitals which are suggected for the same.'
        message2 = '1. Jupiter hospital , Balewadi        2. Life Point, near Sayaji hotel, Wakad'
    if(output==5):
        message1='These symptoms are of Helthy person.'
        message2 = ''
    if(output==6):
        message1='According to the symptoms, you are just feeling low.'
        message2 = 'Have a healthy diet'
    if(output==7):
        message1='These symptoms are of Weak immunity. Please do not panic, below are the hospitals which are suggected for the same.'
        message2 = '1. Jupiter hospital , Balewadi        2. Life Point, near Sayaji hotel, Wakad'
    return render_template('index.html', prediction_text1='{}'.format(message3), prediction_text2='{}'.format(message3))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)