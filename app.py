

import os,json
from flask import Flask, render_template, request,abort,send_file
from credit.constants import *
from credit.entity.default_predictor import CreditData, DefaultPredictor
from credit.logger import get_log_df, logging
from credit.exception import CreditException
from credit.pipeline.pipeline import Pipeline
from credit.config.configuration import configuration
from credit.util import write_yaml_file,read_yaml_file


MODEL_DIR= os.path.join(CURR_DIR,SAVED_MODELS_DIR_NAME)
MODEL_CONFIG_FILE_PATH= os.path.join(CURR_DIR,CONFIG_DIR,"model.yaml")

app=Flask(__name__)

@app.route("/",methods=['GET','POST'])
def index():
    try:
       return render_template('index.html')
    except Exception as e:
        return str(e)

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    try:
        experiment_df = Pipeline.get_experiments_status()
        context = {
            "experiment": experiment_df.to_html(classes='table table-striped col-12')
        }
        return render_template('experiment_history.html', context=context)
    except Exception as e:
        return str(e)

@app.route('/train',methods=['GET','POST'])
def train():
    try:
        message=""
        pipeline = Pipeline(config=configuration())
        if not pipeline.experiment.running_status:
            message = "Training started"
            pipeline.start()
        else:
            message="Training is already in progress"
        context = {
                "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
                "message": message
                    }
        return render_template('train.html',context=context)
    except Exception as e:
        return str(e)


@app.route("/update_model_config",methods=['GET','POST'])
def update_model_config():
    try:
       
        if request.method=='POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'",'"')
            logging.info(model_config)
            model_config = json.loads(model_config)
            
            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH,data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html',result={"model_config":model_config}) 
    except Exception as e:
        logging.exception(e)
        return str(e)

@app.route("/artifact",defaults={'req_path':'credit'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    try:
        #creating artifact dir
        os.makedirs("credit",exist_ok=True)
        logging.info(f"req_path: {req_path}")
        #creating absolute path
        abs_path=os.path.join(req_path)
        logging.info(f"abs_path:{abs_path}")
        #return 404 if path does not exist
        if not os.path.exists(abs_path):
            return abort(404)
        #check if path is a file and save
        if os.path.isfile(abs_path):
            if ".html" in abs_path:
                with open(abs_path,"r",encoding="utf-8") as file:
                    content=""
                    for line in file.readlines():
                        content = f"{content}{line}"
                    return content
            return send_file(abs_path)

        # Show directory contents
        files = {os.path.join(abs_path,file_name): file_name for file_name in os.listdir(abs_path) if 
                "artifact" in os.path.join(abs_path,file_name)}

        result = {
        "files":files,
        "parent_folder":os.path.dirname(abs_path),
        "parent_label": abs_path
            }
        return render_template('files.html',result=result)
    except Exception as e:
        return str(e)

@app.route(f'/logs', defaults={'req_path': 'logs'})
@app.route(f'/logs/<path:req_path>')
def render_log_dir(req_path):
    try:
        os.makedirs("logs", exist_ok=True)
        # Joining the base and the requested path
        logging.info(f"req_path: {req_path}")
        abs_path = os.path.join(req_path)
        logging.info(f"abs_path: {abs_path}")
        # Return 404 if path doesn't exist
        if not os.path.exists(abs_path):
            return abort(404)

        # Check if path is a file and save
        if os.path.isfile(abs_path):
            log_df = get_log_df(abs_path)
            context = {"log": log_df.to_html(classes="table-striped", index=False)}
            return render_template('log.html', context=context)

        # Show directory contents
        files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

        result = {
            "files": files,
            "parent_folder": os.path.dirname(abs_path),
            "parent_label": abs_path
        }
        return render_template('log_files.html', result=result)
    except Exception as e:
        return str(e)

@app.route('/saved_models',defaults={'req_path':'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    try:
        os.makedirs("saved_models",exist_ok=True)
        #Joining the base and the requested path
        logging.info(f"req_path:{req_path}")
        abs_path = os.path.join(req_path)
        logging.info(f"abs_path : {abs_path}")
        #Return 404 if path doesn't exist
        if not os.path.exists(abs_path):
            return abort(404)
        #check if path is a file and save
        if os.path.isfile(abs_path):
            return send_file(abs_path)

        #show directory contents 
        files = {os.path.join(abs_path,file): file for file in os.listdir(abs_path)}

        result = {
            "files":files,
            "parent_folder":os.path.dirname(abs_path),
            "parent_label":abs_path
                }

        return render_template('saved_models_files.html',result=result)
    except Exception as e:
        logging.exception(e)
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