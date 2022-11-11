from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
mdl = pickle.load(open('gpred.pkl','rb'))
data1=pd.read_csv('Dataset88.csv')
data2=pd.read_csv('description77.csv')
data3=pd.read_csv('precaution99.csv')
app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def main():
    return render_template('mlweb.html')
@app.route('/predict',methods=['POST'])
def prediction():
    s1=request.form['sym1']
    s2=request.form['sym2']
    s3=request.form['sym3']
    s4=request.form['sym4']
    s5=request.form['sym5']
    s6=request.form['sym6']
    dt=np.array([s1,s2,s3,s4,s5,s6])
    k=mdl.predict(pd.DataFrame([dt],columns= ['Symptom_1','Symptom_2','Symptom_3','Symptom_4','Symptom_5','Symptom_6']))
    return render_template('mlweb.html',res=k)

if __name__=='__main__':
    app.run(debug=True)
    

