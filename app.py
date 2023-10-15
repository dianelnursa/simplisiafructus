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

model = tf.keras.models.load_model("Xception-fructus-97.29.h5")

st.write(
    """
         # Deteksi Simplisia Fructus
         """
)

st.write("Untuk mendeteksi jenis simplisia fructus berdasarkan bentuknya. Aplikasi saat ini terbatas hanya dapat mendeteksi 7 jenis simplisia fructus  meliputi: \n (1) Amomi Fructus/Kapulaga \n (2) Cumini Fructus/Jinten \n (3) Piperis Albi Ftuctus/Lada Putih \n (4) Piperis Nigri Fructus/Lada Hitam \n (5) Piper Retrofractum Fructus/Cabai Jawa \n (6) Tamarindus Indicia Fructus/Asam Jawa \n (7) Capsici Frutescentis Fructus/ Cabai Rawit.")

file = st.file_uploader("Silahkan upload gambar", type=["jpg", "png"])

if file is None:
    st.text("Belum ada gambar")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    # print("prediction : ",prediction)
    if np.argmax(prediction) == 0:
        st.write("Hasil Terdeteksi: Amomi Fructus/Kapulaga")
        st.write("Indikasi Kegunaan: \n 1. Membantu pencernaan, meredakan perut kembung, mual, dan muntah. \n 2. Mengatasi masalah pernapasan seperti batuk dan sesak napas. \n 3. Memberikan efek relaksasi pada tubuh. \n 4. Memberikan rasa dan aroma khas pada makanan dan minuman.")
    elif np.argmax(prediction) == 1:
        st.write("Hasil Terdeteksi: Capsici Frutescentis Fructus/Cabai Rawit")
    elif np.argmax(prediction) == 2:
        st.write("Hasil Terdeteksi: Cumini Fructus/Jinten")
        st.write("Indikasi Kegunaan: \n 1. Membantu pencernaan dan meredakan masalah perut. \n 2. Digunakan sebagai penyedap alami dalam berbagai hidangan. \n 3. Mengandung senyawa sehat seperti antioksidan. \n 4. Digunakan dalam tradisi pengobatan alternatif untuk beberapa masalah kesehatan.")
    elif np.argmax(prediction) == 3:
        st.write("Hasil Terdeteksi: Piper Retrofractum Fructus/Cabai Jawa")
        st.write("Indikasi Kegunaan: \n 1. Membantu meredakan gejala masuk angin seperti mual, muntah dan perut kembung. \n 2. Membantu meredakan sakit kepala dan nyeri otot \n 3. Meningkatkan nafsu makan dan stimulan kesehatan dan tonik")
    elif np.argmax(prediction) == 4:
        st.write("Hasil Terdeteksi: Piperis Albi Fructus/Lada Putih")
    elif np.argmax(prediction) == 5:
        st.write("Hasil Terdeteksi: Piperis Nigri Fructus/Lada Hitam")
        st.write("Indikasi Kegunaan: \n 1. Penyedap alami dalam masakan. \n 2. Membantu pencernaan dan meredakan perut kembung \n 3. Meredakan masalah pernapasan seperti pilek dan batuk. \n 4. Mengatasi peradangan dalam tubuh. \n 5. Meningkatkan sistem kekebalan tubuh.")
    elif np.argmax(prediction) == 6:
        st.write("Hasil Terdeteksi: Tamarindus Indicia Fructus/Asam Jawa")
        st.write("Indikasi Kegunaan: \n 1. Meredakan masalah pencernaan seperti sembelit dan perut kembung. \n 2. Sumber antioksidan dan vitamin C. \n 3. Pemanis alami dalam makanan dan minuman. \n 4. Bumbu dapur berbagai hidangan kuliner.")
    elif np.argmax(prediction) == 7:
        st.write("Bukan Simplisia Fructus")
