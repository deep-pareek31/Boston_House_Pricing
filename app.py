import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
reg_model=pickle.load(open('Linear_reg_model.pkl','rb'))
scaler=pickle.load(open('Standard_Scaler.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    ##the input we gave in form of jason is captured by 'data' key and when 
    ##when we hit the predict_api ,the data present in 'data' will be captured by reqst.json
    data=request.json['data']  
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=reg_model.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict',methods=["POST"])
def predict():
   data=[float(x) for x in request.form.values()]
   final_input=scaler.transform(np.array(data).reshape(1,-1))
   print(final_input)
   output=reg_model.predict(final_input)[0]
   return render_template("home.html",prediction_text='The House Pice Prediction is {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)