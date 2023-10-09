import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps


def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = image.convert("RGB")
    image = np.asarray(image)
    image = image.astype(np.float32) / 255.0

    img_reshape = image[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


model = tf.keras.models.load_model("Xception-fructus.h5")

st.write(
    """
         # Deteksi Simplisia Fructus
         """
)

st.write("Untuk mendeteksi jenis simplisia fructus berdasarkan bentuk")

file = st.file_uploader("Silahkan upload gambar", type=["jpg", "png"])

if file is None:
    st.text("Belum ada gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    # print("prediction : ",prediction)
    if np.argmax(prediction) == 0:
        st.write("Hasil Terdeteksi: Piperis Nigri Fructus/Lada Hitam")
    elif np.argmax(prediction) == 1:
        st.write("Hasil Terdeteksi: Piperis Albi Fructus/Lada Putih")
    elif np.argmax(prediction) == 2:
        st.write("Hasil Terdeteksi: Piper Retrofractum Fructus/Cabai Jawa")
    elif np.argmax(prediction) == 3:
        st.write("Hasil Terdeteksi: Cumini Fructus/Jinten")
    elif np.argmax(prediction) == 4:
        st.write("Hasil Terdeteksi: Amomi Fructus/Kapulaga")
    elif np.argmax(prediction) == 5:
        st.write("Hasil Terdeteksi: Capsici Frutescentis Fructus/Cabai Rawit")
