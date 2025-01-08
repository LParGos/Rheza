import streamlit as st
import joblib

# Memuat model
svm_model_filename = 'E:/1Tgs kuliah/Machine learning/uas/DatasetSupervised/svm.pkl'
svm_model = joblib.load(svm_model_filename)

lr_model_filename = 'E:/1Tgs kuliah/Machine learning/uas/DatasetSupervised/LR.pkl'
lr_model = joblib.load(lr_model_filename)

# Aplikasi Streamlit
st.title('Prediksi Fruit Berdasarkan Fitur')

# Form input untuk prediksi
with st.form(key='input_form'):
    diameter = st.number_input('Diameter', min_value=0.0)
    berat = st.number_input('Weight', min_value=0.0)
    merah = st.number_input('Red', min_value=0, max_value=255)
    hijau = st.number_input('Green', min_value=0, max_value=255)
    biru = st.number_input('Blue', min_value=0, max_value=255)
    
    # Pilihan algoritma
    algorithm = st.selectbox('Pilih Algoritma', ['SVM', 'Logistic Regression'])
    
    submit_button = st.form_submit_button('Prediksi')

# Prediksi jika form telah disubmit
if submit_button:
    # Membuat array fitur input
    input_data = [[diameter, berat, merah, hijau, biru]]
    
    # Melakukan prediksi berdasarkan pilihan algoritma
    if algorithm == 'SVM':
        prediksi = svm_model.predict(input_data)
    else:
        prediksi = lr_model.predict(input_data)
    
    st.write(f"Nama buah yang diprediksi adalah: {prediksi[0]}")