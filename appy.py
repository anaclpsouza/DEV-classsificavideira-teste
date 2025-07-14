import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np

@st.cache_resource
def carrega_modelo():
    #https://drive.google.com/file/d/1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp/view?usp=sharing
    url = 'https://drive.google.com/uc?id=1GpWix8dp6FeFAs6g0etbnw_avu9Aflfp'

    gdown.download(url, 'modelo_quantizado16bits.tflite')

    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

    interpreter.alocate_tensors()

    return interpreter

def load_image():
    uploaded_file = st.file_uploader("arraste e solte uma imagem ou clique para selecionar", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image)
        st.success("Imagem carregada com sucesso")
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def main():
    st.set_page_config(
        page_title="Classifica Folhas de Videiras!"
    )

    st.write("# Classifica Folhas de Videiras")
     
    interpreter = carrega_modelo()

    image = load_image()


    
if __name__ == "__main__": main()