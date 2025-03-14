import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
import numpy as np
import time

st.title("Coffee Prediction ☕")

df = pd.read_csv("dataset/arabicadata.csv")
st.subheader("ตัวอย่างของชุดข้อมูล")

df = df[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
         "Uniformity", "Clean.Cup", "Sweetness", "Country.of.Origin"]]

numeric_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
                "Uniformity", "Clean.Cup", "Sweetness"]
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

st.dataframe(df.head())

country_col = df["Country.of.Origin"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
scaled_df["Country.of.Origin"] = country_col.values

country = LabelEncoder()
country.fit(df["Country.of.Origin"])

st.subheader("ประเทศที่มีอยู่ในชุดข้อมูล")
st.write(np.array(country.classes_).reshape(1, -1))

usedata = df.drop(columns=["Country.of.Origin"])
finedata = df["Country.of.Origin"]

encoder = LabelEncoder()
finedata = encoder.fit_transform(finedata)

traindata, testdata, trantarget, testtarget = train_test_split(
    usedata, finedata, test_size=0.1, random_state=42)

imputer = SimpleImputer(strategy="mean")
traindata = imputer.fit_transform(traindata)

scaler = StandardScaler()
traindata = scaler.fit_transform(traindata)

model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                      alpha=0.001, max_iter=100, random_state=42)
model.fit(traindata, trantarget)

st.subheader("มาหาแหล่งที่มาของกาแฟกันดีกว่า")
st.text("มาใส่ข้อมูลของกาแฟกัน (โดยข้อมูลที่กรอกคือ คะแนนระหว่าง 0.00 - 10.00)")

fetcol = usedata.columns

inputs = {}
for col in fetcol:
    inputs[col] = st.number_input(f"{col}", value=5.0, step=0.1)

if st.button("🔎 หาที่มาของกาแฟกัน"):
    progress_bar = st.progress(0)
    for percent in range(0, 101, 10):
        time.sleep(0.2)
        progress_bar.progress(percent)

    input = np.array([[inputs[col] for col in fetcol]])
    input = imputer.transform(input)
    input = scaler.transform(input)

    prediction_probs = model.predict_proba(input)[0]
    prediction = np.argmax(prediction_probs)

    country = encoder.inverse_transform([prediction])[0]
    confidence = prediction_probs[prediction] * 100

    st.success(f"แหล่งที่มาของกาแฟคือ : {country}")
    st.success(f"ความมั่นใจในการทำนายคือ : {confidence:.2f} %")
