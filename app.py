import sys
from flask import Flask, render_template
from credit.logger import logging
from credit.exception import CreditException
HOUSING_DATA_KEY = "housing_data"
MEDIAN_HOUSING_VALUE_KEY = "median_house_value"

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    try:
       return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route("/predict",methods=['GET','POST'])
def predict():
    try:
        context = {
            HOUSING_DATA_KEY:None,
            MEDIAN_HOUSING_VALUE_KEY:None
                    }
        return render_template("predict.html",context=context)
    except Exception as e:
       return str(e)


if __name__=="__main__":
    app.run(debug=True)