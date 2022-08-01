from setuptools import setup,find_packages
from typing import List
#Declaring varibales for the setup function
PROJECT_NAME="Credict_default_predictor"
VERSION="0.0.1"
AUTHOR="SOMESH_TRIVEDI"
DESCRIPTION= "MACHINE LEARNING PROJECT TO PREDICT CREDIT DEFAULTER"

REQUIREMENT_FILE=  "requirements.txt"

HYPHEN_E_DOT="-e ."

def get_requirements_list()->List[str]:
    """Description: This function is going to return list of requirement
    mention in requirements.txt file
    return This function is going to return a list which contain name
    of libraries mentioned in requirements.txt file"""
    with open(REQUIREMENT_FILE) as req:
        req_list = req.readlines()
        if HYPHEN_E_DOT in req_list :
            req_list.remove(HYPHEN_E_DOT)
        return req_list
        


setup(name=PROJECT_NAME,
    description=DESCRIPTION,
    version=VERSION,
    author=AUTHOR,
    packages=find_packages(),
    install_requires=get_requirements_list()
    )
