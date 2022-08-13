

import os
from flask import Flask, render_template, request,abort,send_file
from credit.constants import *
from credit.entity.default_predictor import CreditData, DefaultPredictor
from credit.logger import logging
from credit.exception import CreditException


MODEL_DIR= os.path.join(CURR_DIR,SAVED_MODELS_DIR_NAME)


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
           CREDITDATAKEY:None,
            DEFAULT_PAYMENT_NEXT_MONTH_KEY:None
                    }
        if request.method=='POST':
            ID = int(request.form['ID'])
            LIMIT_BAL=float(request.form['LIMIT_BAL'])
            SEX = int(request.form['SEX'])
            EDUCATION = int(request.form['EDUCATION'])
            MARRIAGE = int(request.form['MARRIAGE'])
            AGE = int(request.form['AGE'])
            PAY_0 = int(request.form['PAY_0'])
            PAY_2= int(request.form['PAY_2'])
            PAY_3= int(request.form['PAY_3'])
            PAY_4= int(request.form['PAY_4'])
            PAY_5= int(request.form['PAY_5'])
            PAY_6= int(request.form['PAY_6'])
            BILL_AMT1 = float(request.form['BILL_AMT1'])
            BILL_AMT2= float(request.form['BILL_AMT2'])
            BILL_AMT3= float(request.form['BILL_AMT3'])
            BILL_AMT4= float(request.form['BILL_AMT4'])
            BILL_AMT5= float(request.form['BILL_AMT5'])
            BILL_AMT6= float(request.form['BILL_AMT6'])
            
            PAY_AMT1= float(request.form['PAY_AMT1'])
            PAY_AMT2= float(request.form['PAY_AMT2'])
            PAY_AMT3= float(request.form['PAY_AMT3'])
            PAY_AMT4= float(request.form['PAY_AMT4'])
            PAY_AMT5= float(request.form['PAY_AMT5'])
            PAY_AMT6= float(request.form['PAY_AMT6'])

            credit_data=CreditData(ID=ID,
                                LIMIT_BAL=LIMIT_BAL,
                                SEX=SEX,
                                
                                AGE=AGE,
                                EDUCATION=EDUCATION,
                                MARRIAGE=MARRIAGE,
                                PAY_0=PAY_0,
                                PAY_2=PAY_2,
                                PAY_3=PAY_3,
                                PAY_4=PAY_4,
                                PAY_5=PAY_5,
                                PAY_6=PAY_6,
                                PAY_AMT1=PAY_AMT1,
                                PAY_AMT2=PAY_AMT2,
                                PAY_AMT3=PAY_AMT3,
                                PAY_AMT4=PAY_AMT4,
                                PAY_AMT5=PAY_AMT5,
                                PAY_AMT6=PAY_AMT6,
                                BILL_AMT1=BILL_AMT1,
                                BILL_AMT2= BILL_AMT2,
                                BILL_AMT3= BILL_AMT3,
                                BILL_AMT4= BILL_AMT4,
                                BILL_AMT5= BILL_AMT5,
                                BILL_AMT6= BILL_AMT6)
            
            credit_df=credit_data.get_default_input_dataframe()
            default_predictor=DefaultPredictor(model_dir=MODEL_DIR)
            default_payment_next_month =default_predictor.predict(X=credit_df)
            
            if default_payment_next_month==0:
                output="Person will not default"
            else:
                output= "Person will default"
            context={
                    CREDITDATAKEY:credit_data.get_default_data_as_dict(),
                    DEFAULT_PAYMENT_NEXT_MONTH_KEY:output}
            return render_template("predict.html",context=context)
            

        return render_template("predict.html",context=context)
    except Exception as e:
       return str(e)


if __name__=="__main__":
    app.run(debug=True)