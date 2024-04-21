from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

# __name__ is the entry point of our application.

app = application

# create Route for our Home page

@app.route('/')
def index():
    return render_template('index.html') # create index.html file in templates folder 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # in this part, we get the data and then predict.
    if request.method=='GET':
        return render_template('home.html') # home.html will have data fields which are used provide inputs.
        # create home.html file in templates folder 
    else:
        # we start creating our data. For that we create our own custom class.
        # We create Same CustomData class in predict_pipeline.py   
        
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        # Now, once we get all the data, we convert it to DataFrame.
        # For this we call get_data_as_data_frame() function
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline =  PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        # results is in list format. Like [65.75]
        return render_template('home.html', results=results[0])


if __name__=="__main__":
    app.run(host="0.0.0.0")     #"0.0.0.0" is going to map with 127.0.0.1:5000/
                                            # 5000 is the default port