import numpy as np
import pandas as pd
from flask import Flask,request,render_template
# import pickle
from sklearn.externals import  joblib
import traceback

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['post'])
def predict():
    try:
        mod = joblib.load('iris_model.sav')
        input_features=[float(x) for x in request.form.values()]
        feature_values=[np.array(input_features)]

        # feature_names=['sepal_length','sepal_width','petal_length','petal_width']
        #
        # df=pd.DataFrame(feature_values,columns=feature_names)
        # output=mod.predict(df)
        output = mod.predict(feature_values)

        return render_template('index.html',prediction_text='the flower is {}'.format(output[0]))
    except Exception as e:
        message = "exception \n" + str(traceback.format_exc())
        return message

if __name__=="__main__":
    app.run()
