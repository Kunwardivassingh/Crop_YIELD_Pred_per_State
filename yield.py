from flask import Flask,request,render_template
import numpy as np
import pickle
import sklearn
print(sklearn.__version__)
#laoding models
dtre=pickle.load(open('dtre.pkl','rb'))
preprocessor=pickle.load(open('preprocessor.pkl','rb'))

#flask app
app= Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    # gettinf input values from the form
    if request.method =='POST':
        Crop_Year=float(request.form['Crop_Year'])
        avg_annual_rainfall_in_mm=float(request.form['avg_annual_rainfall_in_mm'])
        Pesticide=float(request.form['Pesticide'])
        Fertilizer=float(request.form['Fertilizer'])
        State=request.form['State']
        Area=float(request.form['Area'])
        Crop=request.form['Crop']
        
        features=np.array([[Crop_Year,avg_annual_rainfall_in_mm,Pesticide,Fertilizer,State,Area,Crop]],dtype=object)
        transform_features=preprocessor.transform(features)
        predicted_yield=dtre.predict(transform_features).reshape(-1,1)
    
        return render_template('index.html',predicted_yield=predicted_yield[0][0])

if __name__=="__main__":
    app.run(debug=True)