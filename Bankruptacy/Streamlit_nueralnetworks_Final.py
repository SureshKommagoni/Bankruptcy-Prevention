
import numpy as np
import pickle
import pandas as pd
import streamlit as st

from PIL import Image

pickle_in = open("D:/DATA SCIENCE/ExcelR/Live Projects/Bankrupt prevention/nueral_mlp_model.pkl","rb")
classifier=pickle.load(pickle_in)


def predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk):
     input = np.array([[industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk]]).astype(np.float64)
     prediction = classifier.predict(input)
     return float(prediction)

def main():
    st.title("Bankruptcy Detector")
    html_temp = """
    <div style="background-color:#0CDB08;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bankruptcy Detector ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    industrial_risk = st.text_input("industrial_risk","Type Here")
    management_risk = st.text_input("management_risk","Type Here") 
    financial_flexibility = st.text_input("financial_flexibility","Type Here")
    credibility = st.text_input("credibility","Type Here")
    competitiveness = st.text_input("competitiveness","Type Here")
    operating_risk = st.text_input("operating_risk","Type Here")
    result=""
    if st.button("Predict"):
        result=predict_bankruptcy(industrial_risk,management_risk,financial_flexibility,credibility,competitiveness,operating_risk)
        st.success('The output is {}'.format(result))
    

if __name__=='__main__':
    main()