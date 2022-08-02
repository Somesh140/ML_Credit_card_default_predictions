import sys
from flask import Flask
from credit.logger import logging
from credit.exception import CreditException

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def index():
    try:
        raise Exception("We are testing custom exception")
    except Exception as e:
        credit = CreditException(e,sys)
        logging.info(credit.error_message)
    logging.info("we have logged")
    return 'ml-project'

if __name__=="__main__":
    app.run(debug=True)