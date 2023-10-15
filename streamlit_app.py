from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image
#import tensorflow as tf

"""
Plant Leaf disease identification using CNN model

"""


with st.echo(code_location='below'):
    total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
    num_turns = st.slider("Number of turns in spiral", 1, 100, 9)

    Point = namedtuple('Point', 'x y')
    data = []

    points_per_turn = total_points / num_turns

    for curr_point_num in range(total_points):
        curr_turn, i = divmod(curr_point_num, points_per_turn)
        angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
        radius = curr_point_num / total_points
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        data.append(Point(x, y))

    st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
        .mark_circle(color='#0068c9', opacity=0.5)
        .encode(x='x:Q', y='y:Q'))



@st.cache(allow_output_mutation=True)

def load_image(image_file):
	img = Image.open(image_file)
	return img

def predict(model,image):
  prediction = model.predict(image)
  return prediction

st.title("Plant Disease Identification")
#model=load_model()
menu = ["Upload","Camera"]
menuModel = ["Load","API"]
# choice = st.sidebar.selectbox("Menu",menu)


choice=st.sidebar.radio('Options', options=menu)

if choice == "Upload":
    st.subheader("Upload")
   # modelChoice = st.radio('Model Options', options=menuModel)

    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])



    if image_file is not None:
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        imageLoad=load_image(image_file)
        st.image(imageLoad,width=250)
        #st.button('Predict')
        if st.button('Predict'):
          st.write ("Test Prediction")
#prediction = predict(model,imageLoad)


if choice == "Camera":
    st.subheader("Take Picture")
    picture = st.camera_input("Take a picture")

    if picture is not None:
        st.image(picture)