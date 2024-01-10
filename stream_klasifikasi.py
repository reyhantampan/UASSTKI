import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

model_klasifikasi_Label = pickle.load(open('Model/model_klasifikasi_NB_Label.sav', 'rb'))
model_klasifikasi_Subreddit = pickle.load(open('Model/model_klasifikasi_Forest_Subreddit.sav', 'rb'))

tfidf = TfidfVectorizer()

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("Model/new_selected_feature_tf-idf.sav", "rb"))))

st.title('Prediksi Gangguan Stressing')

clean_teks = st.text_area('Masukkan Teks', height=200)

Label = ''
Subreddit = ''

if st.button('Deteksi'):
    clean_teks_list = [clean_teks]
    
    loaded_vec.fit_transform(clean_teks_list)
    
    transformed_text = loaded_vec.transform(clean_teks_list)
    
    dense_transformed_text = transformed_text.toarray()
    
    predict_labeling = model_klasifikasi_Label.predict(dense_transformed_text)
    
    if predict_labeling == 0:
        Label = 'No Stress'
    elif predict_labeling == 1:
        Label = 'Stress'
    else:
        Label = 'Netral'
    
    predict_subreddit = model_klasifikasi_Subreddit.predict(dense_transformed_text)
    
    if predict_labeling == 1:
        Subreddit = ', '.join(predict_subreddit.tolist())

        df = pd.DataFrame(predict_subreddit, columns=['Subreddit'])
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.countplot(x='Subreddit', data=df, ax=ax)
        st.pyplot(fig)

    else:
        Subreddit = "Baik baik Saja"

st.success(Label)
st.success(Subreddit)
