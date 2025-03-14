import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.title("🌍 The Earthquake")

df = pd.read_csv("dataset/Earthquakesdataset.csv")
st.write("ตัวอย่างข้อมูล:")
st.dataframe(df.head())

features = ["depth_km", "longitude", "latitude", "sig", "gap", "mag"]
missing_values = df.isnull().sum()
df_cleaned = df.fillna(0)


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


original_size = df_cleaned.shape[0]
for feature in features:
    if feature in df_cleaned.columns:
        df_cleaned = remove_outliers(df_cleaned, feature)

features_model = ["depth_km", "longitude", "latitude", "sig", "gap"]
target = "mag"

X = df_cleaned[features_model]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

svmmodel = SVR()
svmmodel.fit(X_train, y_train)
svmpredic = svmmodel.predict(X_test)

knnmodel = KNeighborsRegressor(n_neighbors=5)
knnmodel.fit(X_train, y_train)
knnpredic = knnmodel.predict(X_test)

predicdata = X_test.copy()
predicdata["SVMpredic"] = svmpredic
predicdata["KNNpredic"] = knnpredic
predicdata["Actual_Mag"] = y_test.values

model_choice = st.selectbox("โมเดลที่ต้องการ", ("SVM", "KNN"))

top_n = st.slider(
    "เลือกว่าจะดูพื้นที่ ที่มีความเสี่ยงมากกี่พื้นที่:", 1, 30, 10)

if model_choice == "SVM":
    st.subheader(
        "พื้นที่ ที่มีความเสี่ยงในการเกิดแผ่นดินไหวมากที่สุดของ (SVM)")
    st.dataframe(predicdata[["depth_km", "longitude", "latitude", "sig",
                             "gap", "SVMpredic", "Actual_Mag"]].nlargest(top_n, "SVMpredic"))

    def evaluate_model(name, y_true, y_pred):
        Error = mean_absolute_error(y_true, y_pred)
        Errorpercent = (Error / y_true.mean()) * 100

        Accuracy = r2_score(y_true, y_pred)
        Accuracypercent = Accuracy * 100

        st.subheader(f"ประสิทธิภาพในการทำนายของ {name}")
        st.write(f"ค่าความคลาดเคลื่อน: {Errorpercent:.2f} %")
        st.write(f"ค่าความแม่นยำ: {Accuracypercent:.2f} %")

    evaluate_model("SVM ", y_test, svmpredic)

    def plot_comparison_svm(y_true, svmpredic):
        plt.figure(figsize=(6, 6))

        plt.scatter(y_true, svmpredic, color='pink',
                    alpha=0.6, label="SVM Predictions")
        plt.scatter(y_true, y_true, color='#87CEEB',
                    alpha=0.6, label="Actual Magnitude")
        plt.plot([y_true.min(), y_true.max()], [
                 y_true.min(), y_true.max()], 'k--', lw=2)

        plt.xlabel("Actual Magnitude")
        plt.ylabel("SVM Predicted Magnitude")
        plt.title("SVM Prediction vs Actual")
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    plot_comparison_svm(y_test, svmpredic)

elif model_choice == "KNN":
    st.subheader(
        "พื้นที่ ที่มีความเสี่ยงในการเกิดแผ่นดินไหวมากที่สุดของ (KNN)")
    st.dataframe(predicdata[["depth_km", "longitude", "latitude", "sig",
                             "gap", "KNNpredic", "Actual_Mag"]].nlargest(top_n, "KNNpredic"))

    def evaluate_model(name, y_true, y_pred):
        Error = mean_absolute_error(y_true, y_pred)
        Errorpercent = (Error / y_true.mean()) * 100

        Accuracy = r2_score(y_true, y_pred)
        Accuracypercent = Accuracy * 100

        st.subheader(f"ประสิทธิภาพในการทำนายของ {name}")
        st.write(f"ค่าความคลาดเคลื่อน: {Errorpercent:.2f} %")
        st.write(f"ค่าความแม่นยำ: {Accuracypercent:.2f} %")

    evaluate_model("KNN ", y_test, knnpredic)

    def plot_comparison_knn(y_true, knn_preds):
        plt.figure(figsize=(6, 6))

        plt.scatter(y_true, knn_preds, color='#87CEEB',
                    alpha=0.6, label="KNN Predictions")
        plt.scatter(y_true, y_true, color='pink',
                    alpha=0.6, label="Actual Magnitude")
        plt.plot([y_true.min(), y_true.max()], [
                 y_true.min(), y_true.max()], 'k--', lw=2)

        plt.xlabel("Actual Magnitude")
        plt.ylabel("KNN Predicted Magnitude")
        plt.title("KNN Prediction vs Actual")
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    plot_comparison_knn(y_test, knnpredic)
