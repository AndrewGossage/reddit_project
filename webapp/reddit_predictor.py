"""parts of this code were adapted from a dsi lesson
"""

import matplotlib.pyplot as plt 
import numpy as np
import pickle 
import pandas as pandas
import streamlit as st 
import streamlit.components.v1 as components

st.set_page_config(
    page_icon='',
    initial_sidebar_state = 'expanded'
)
@st.cache(allow_output_mutation=True)
def load_pickles():
    with open('../pickels/randomforest.pkl', mode ='rb') as pickle_in:
        rf = pickle.load(pickle_in).best_estimator_
    
    with open('../pickels/cv.pkl', mode ='rb') as pickle_in:
        cv = pickle.load(pickle_in)
    
    return rf, cv
rf,cv = load_pickles()

page = st.sidebar.selectbox(
    'Page',
    ('About','EDA', 'Make a Prediction')
)


if page == 'About':
    with open('about.html', mode ='r') as about:
        components.html(about.read(), height = 2000)

if page == 'EDA':
    with open('eda.html', mode ='r') as about:
        components.html(about.read(), height = 2500)
    
elif page == 'Make a Prediction':
    st.title('Dark Souls or Sekiro')
    st.write('Enter text related to Dark Souls or Sekiro: Shadows Die Twice:')

    your_text = st.text_area('please enter a paragraph of text.')
    grid = cv.transform([your_text])
    predicted_author = rf.predict(cv.transform([your_text.lower()]))
    dictio = {0: 'Sekiro Shadows Die Twice', 1:'Dark Souls'}
    if len(your_text) > 0:
        st.write(f'This post is about {dictio[predicted_author[0]]}.')
    
    


