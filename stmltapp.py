
import numpy as np
import pickle
import pandas as pd

import streamlit as st 

from PIL import Image


pickle_in = open("credit.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"


def predict_creditscore(recency,frequency,monetary):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict([[recency,frequency,monetary]])
    print(prediction)
    return prediction



def main():
    st.title("Credit Scoring Analysis")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Credit Scoring ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    recency = st.text_input("Recency","Type Here")
    frequency = st.text_input("Frequency","Type Here")
    monetary = st.text_input("Monetary","Type Here")
    
    result=""
    if st.button("Predict"):
        result=predict_creditscore(recency,frequency,monetary)
    st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()
    
    
    
    
    
    