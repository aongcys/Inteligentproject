import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
import numpy as np
import time

st.title("Coffee Prediction ☕")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset/arabicadata.csv")
    st.subheader("ตัวอย่างของชุดข้อมูล")
    st.dataframe(df.head())
    df = df[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance", 
             "Uniformity", "Clean.Cup", "Sweetness", "Country.of.Origin"]]
    
    df.dropna(inplace=True)
    return df

df = load_data()

country = LabelEncoder()
country.fit(df["Country.of.Origin"])

st.subheader("ประเทศที่มีอยู่ในชุดข้อมูล")
st.write(np.array(country.classes_).reshape(1, -1))

st.subheader("มาหาแหล่งที่มาของกาแฟกันดีกว่า")
st.text("มาใส่ขอมูลของกาแฟกัน (โดยข้อมูลที่กรอกคือ คะแนนระหว่าง 0.00 - 10.00)")


inputs = {}
for col in df.columns[:-1]: 
    inputs[col] = st.number_input(f"{col}", value=5.0, step=0.1)
    
if st.button("🔎 ตรวจสอบที่มาของกาแฟ"):
    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.2)
        progress_bar.progress(percent)
    
    usedata = df.drop(columns=["Country.of.Origin"])
    finedata = df["Country.of.Origin"]
    
    encoder = LabelEncoder()
    finedata = encoder.fit_transform(finedata)

    traindata, testdata,trantarget, testtarget = train_test_split(usedata, finedata, test_size=0.3, random_state=42)

    imputer = SimpleImputer(strategy="mean")
    traindata = imputer.fit_transform(traindata)
    testdata = imputer.transform(testdata)

    scaler = StandardScaler()
    traindata = scaler.fit_transform(traindata)
    testdata = scaler.transform(testdata)

    mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                   alpha=0.01, max_iter=1000, random_state=42)
    mlp_classifier.fit(traindata, trantarget)

    input_values = np.array([[inputs[col] for col in usedata.columns]])
    input_values = imputer.transform(input_values)
    input_values = scaler.transform(input_values)

    prediction_probs = mlp_classifier.predict_proba(input_values)[0]
    prediction = np.argmax(prediction_probs) # เลือกประเทศที่มีค่าความน่าจะเป็นมากสุด
    predicted_country = encoder.inverse_transform([prediction])[0] # แปลงค่าตัวเลขกลับเป็นชื่อประเทศ
    confidence = prediction_probs[prediction] * 100
    
    st.success(f"แหล่งที่มาของกาแฟคือ : {predicted_country}")
    st.success(f"ความมั่นใจในการทำนายคือ : {confidence:.2f} %")
