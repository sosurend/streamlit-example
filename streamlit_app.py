
import streamlit as st
from PIL import Image
from PIL import UnidentifiedImageError
import tensorflow as tf
import numpy as np
import os
import time
from gtts import gTTS
from io import BytesIO
from keras.preprocessing.image import ImageDataGenerator
from googletrans import Translator
translator = Translator()
import emoji

#-----------------------------------------------------
from streamlit_feedback import streamlit_feedback
import datetime
x = datetime.datetime.now()
FEEDBACK_FILE = "/content/drive/MyDrive/CapstoneGroup6/Feedback/fb" + str(x.year) + str(x.month) + str(x.day) + ".csv"

def _submit_feedback(user_response, emoji=None):
    st.toast(f"Feedback submitted: {user_response}", icon=emoji)
    return user_response.update({"time": int(time.time())})


feedback_kwargs = {
        "feedback_type": "thumbs", # "faces"
        "optional_text_label": "Please provide feedback",
        "on_submit": _submit_feedback,
    }

if "feedback_key" not in st.session_state:
  st.session_state.feedback_key = 0

ufb = streamlit_feedback(
                **feedback_kwargs,
                #key=feedback_key,
            )
if ufb:
  #st.write(":orange[Component output:]")
  #st.write(ufb["text"])
  if emoji.demojize(ufb["score"]) == ":thumbs_down:":
    ufb["text"].replace(","," ")
    csvstr = str(ufb["time"]) + "," + "thumbs_down" + "," + ufb["text"] + "\n"
  elif emoji.demojize(ufb["score"]) == ":thumbs_up:":
    ufb["text"].replace(","," ")
    csvstr = str(ufb["time"]) + "," + "thumbs_up" + "," + ufb["text"] + "\n"

  fbfile = open(FEEDBACK_FILE, "a")
  fbfile.write(csvstr)
  fbfile.close()


# ufb is {
#  "type": "thumbs",
#  "score": "👎",
#  "text": "not good",
#  "time" : ""
#}

#-----------------------------------------------------

UPLOAD_DIR="/content/upload_dir/"
INFO_DIR="/content/drive/MyDrive/CapstoneGroup6/PlantInfo/"

plantinfo_files = {
  'Apple___Apple_scab' : "AppleScab.txt",
  'Apple___Black_rot' : "AppleBlackRot.txt",
  'Apple___Cedar_apple_rust' : "AppleCedarRust.txt",
  "Cherry___Powdery_mildew" : "CherryPowderyMildew.txt",
  'Corn___Cercospora_leaf_spot Gray_leaf_spot' : "CornCercosporaLeafSpotGrayLeafSpot.txt",
  'Corn___Common_rust' : "CornCommonRust.txt",
  'Corn___Northern_Leaf_Blight' : "CornNorthernLeafBlight.txt",
  'Grape___Black_rot' : "GrapeBlackRot.txt",
  'Grape___Esca_(Black_Measles)' : "GrapeEsca_Black_Measles.txt",
  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)' : "GrapeLeafBlight_Isariopsis_LeafSpot.txt",
  "Orange___Haunglongbing_(Citrus_greening)" : "OrangeHuanglongbing.txt",
  'Peach___Bacterial_spot' : "PeachBacterialSpot.txt",
  'Pepper,_bell___Bacterial_spot' : "BellPepperBacterialSpot.txt",
  'Potato___Early_blight' : "PotatoEarlyBlight.txt",
  'Potato___Late_blight' : "PotatoLateBlight.txt",
  'Squash___Powdery_mildew' : "SquashPowderyMildew.txt",
  "Strawberry___Leaf_scorch" : "StrawberryLeafScorch.txt",
  'Tomato___Bacterial_spot' : "TomatoBacterialSpot.txt",
  'Tomato___Early_blight' : "TomatoEarlyBlight.txt",
  'Tomato___Late_blight' : "TomatoLateBlight.txt",
  'Tomato___Leaf_Mold' : "TomatoLeafMold.txt",
  "Tomato___Septoria_leaf_spot" : "TomatoSeptoriaLeafSpot.txt",
  'Tomato___Spider_mites Two-spotted_spider_mite' : "TomatoSpiderMites_TwoSpottedSpiderMite.txt",
  'Tomato___Target_Spot' : "TomatoTargetSpot.txt",
  'Tomato___Tomato_Yellow_Leaf_Curl_Virus' : "TomatoYellowLeafCurlVirus.txt",
  'Tomato___Tomato_mosaic_virus' : "TomatoMosaicVirus.txt",
  "Tomato___Leaf_Mold" : "TomatoLeafMold.txt"
}
predstart = time.time()
predend = time.time()

@st.cache_resource
def load_model():
  model=tf.keras.models.load_model('/content/drive/MyDrive/CapstoneGroup6/Models/VGG_model.keras')
  return model

def load_image(image_file):
  try:
    img = Image.open(image_file)  
  except UnidentifiedImageError:
    e = RuntimeError("This image format could not be identified. Please upload jpg/jpeg/png only")
    st.exception(e)
  return img

def predict(model,image):
  predstart = time.time()
  prediction = model.predict(image)
  predend = time.time()
  #labels39 = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy','Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
  #        'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
  #        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy','Soybean___healthy',
  #        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
  labels = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves',
          'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust',
          'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
          'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Squash___Powdery_mildew',
          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
          'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
  if prediction.argmax() < len(labels):
    predlabel = labels[prediction.argmax()]
    predprob = int(prediction[0][prediction.argmax()]*100)
  else:
    st.write("debug : prediction is ", prediction)
    predlabel = "Sorry, app error." + str(prediction.argmax())
    predprob = int(prediction[0][prediction.argmax()]*100)

  return predlabel, predprob

def showDiseaseInfo(disease):
  info_file = plantinfo_files.get(disease)
  if info_file:
    #st.write(INFO_DIR + "/" + plantinfo_files[disease])
    info_file = open(INFO_DIR + "/" + plantinfo_files[disease], 'r')
    lines = info_file.readlines()
    info_file.close()
    return lines
  else:
    return ["Sorry, no details available for this disease."]


def prepImage(image_file):
  img = load_image(image_file)
  img.save(UPLOAD_DIR + "images/" + image_file.name)
  gen = ImageDataGenerator(rescale = 1.0/255,
        width_shift_range = 0.005, height_shift_range = 0.005, rotation_range = 0, horizontal_flip = True)
  test_batch = gen.flow_from_directory(
    UPLOAD_DIR,
    target_size=(256, 256),
    color_mode='rgb',
    save_format='png',
    interpolation='nearest',
    keep_aspect_ratio=False
  )
  return next(test_batch)[0]

def playAudio(mytext, tolanguage='en'):
  audio_gttsobj = gTTS(text=mytext, lang=tolanguage, slow=False) # 'en'
  fp = BytesIO()
  audio_gttsobj.write_to_fp(fp)
  fp.seek(0)
  st.audio(fp, format='audio/mp3')

#--------------------------------------
lang_options = {
        "English (US)":"en",  # en_US
        #"Japanese 日本語":"ja_JP",
        "Chinese(simplified) 中文" : "zh-CN",
        "Kannada ಕನ್ನಡ" : "kn",
        "हिंदी" : "hi",
        "Malayalam മലയാളം" : "ml",
        "Telugu తెలుగు" : "te"
    }
locale = st.radio(label='Language', options=list(lang_options.keys()))
g_lang = lang_options[locale]

def transtext(tstr):
  translated_text = translator.translate(tstr, dest=g_lang)
  return translated_text.text


#### Main #####
translated_text = translator.translate("Plant Leaf Disease Identification", dest=g_lang)

st.subheader("Plant Leaf Disease Identification")
if g_lang is not "en": # en_US
  st.subheader(translated_text.text)

col1, col2, col3 = st.columns([3,6,3])
with col1:
  st.title("Dr.Green")

with col2:
  st.image('DrHealthyLeaf.jpeg', width=200)

with col3:
  st.write("")



model=load_model()
menu = ["Upload","Camera"]
menuModel = ["Load","API"]
choice=st.sidebar.radio('Options', options=menu)

str1 = "Supported plants : Tomato, Orange, Strawberry, Apple, Grape, Cherry, Corn, Potato, Capsicum, Peach"
st.subheader(transtext(str1))

if choice == "Upload":
    image_file = st.file_uploader(transtext("Upload a leaf image"), type=["png","jpg","jpeg"])

if choice == "Camera":
    image_file = st.camera_input("Focus on Leaf")

if image_file is not None:
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        imageLoad=load_image(image_file)
        st.image(imageLoad,width=250)
        predict_image = prepImage(image_file)
        if st.button(transtext("Diagnose")):
          predlabel,predprob = predict(model,predict_image)
          predlabel_str = predlabel.replace("_", " ")
          #st.write("Prediction latency : ", str(1000*(predend - predstart)), " milliseconds")

          info=()
          if predlabel is 'Background_without_leaves':
              predstr = transtext("Sorry, leaf could not be identified. Please try uploading clearer pic of a leaf")
              st.write (predstr)
              playAudio(predstr, g_lang)
          elif "healthy" in predlabel.lower():
              tmpstr = predlabel.lower()
              tmpstr = tmpstr.replace("_", "")
              tmpstr = tmpstr.replace("healthy", "")
              predstr = transtext("Healthy & Happy " + tmpstr + " Leaf !!")
              st.write (predstr)
              playAudio(predstr, g_lang)
          elif predprob < 92:
              predstr = transtext("Sorry, Unable to diagnose the exact disease. This could be " + predlabel_str + " with " + str(predprob) + "% probability")
              st.write (predstr)
              playAudio(predstr, g_lang)
          else:
              predstr = predlabel_str + " (with " + str(predprob) + "% probability)"
              st.write (predstr)
              if g_lang != "en":
                st.write (transtext(predstr))
              playAudio(transtext(predstr), g_lang)
              st.divider()
              st.subheader(transtext("Read About " + predlabel_str))
              info = showDiseaseInfo(predlabel)
              trlines = ""
              for line in info:
                trline = transtext(line)
                st.write(trline)
                trlines = trlines + "." + trline
              if trlines is not "":
                playAudio(transtext(trlines), g_lang)
        # end of diagnose
        os.remove(UPLOAD_DIR + "images/" + image_file.name)
