from numpy import where
import joblib
import pandas as pd
from flask import Flask,request
from flask_cors import CORS
import json



app = Flask(__name__)
CORS(app)
filename = 'finalized_model.sav'
loaded_model = joblib.load(filename)

@app.route('/')
def index():
	return "You are at Index" #Just for Testing


@app.route('/predict',methods=['POST'])
def predict():
	event = json.loads(request.data)
	data = event['values']
	print(data)
	#data = [[drugno, dose,age,weight]]
	df = pd.DataFrame(data,columns=['drugno','dose','age','weight'],dtype=float)
	pred = loaded_model.predict(df)
	print(pred)
	return str(pred[0])

if __name__ == '__main__':
	app.run()

