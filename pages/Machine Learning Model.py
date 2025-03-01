import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

st.title("🌍 The Earthquake")

df = pd.read_csv("dataset/Earthquakesdataset.csv")

features = ["depth_km", "longitude", "latitude", "sig", "gap"]
target = "mag"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)

X_test["SVM_Pred"] = svm_preds
X_test["KNN_Pred"] = knn_preds
X_test["Actual_Mag"] = y_test.values

model_choice = st.selectbox("Please select a model to display", ("SVM", "KNN"))

top_n = st.slider("Please select a number of top risk areas:", 1, 30, 10)

if model_choice == "SVM":
    st.subheader("พื้นที่ ที่มีความเสี่ยงในการเกิดแผ่นดินไหวมากที่สุด (SVM)")
    st.dataframe(X_test[["depth_km", "longitude", "latitude", "sig", "gap", "SVM_Pred", "Actual_Mag"]].nlargest(top_n, "SVM_Pred"))
    
    def evaluate_model(name, y_true, y_pred):
        Error = mean_absolute_error(y_true, y_pred)
        Errorpercent = (Error / y_true.mean()) * 100

        Accuracy = r2_score(y_true, y_pred)
        Accuracypercent = Accuracy * 100

        st.subheader(f"ประสิทธิภาพในการทำนายของ {name}")
        st.write(f"ค่าความคลาดเคลื่อน: {Errorpercent:.2f} %")
        st.write(f"ค่าความแม่นยำ: {Accuracypercent:.2f} %")
    
    evaluate_model("SVM ", y_test, svm_preds)

    def plot_comparison_svm(y_true, svm_preds):
        plt.figure(figsize=(6, 6))

        plt.scatter(y_true, svm_preds, color='pink', alpha=0.6, label="SVM Predictions")
        plt.scatter(y_true, y_true, color='#87CEEB', alpha=0.6, label="Actual Magnitude")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        
        plt.xlabel("Actual Magnitude")
        plt.ylabel("SVM Predicted Magnitude")
        plt.title("SVM Prediction vs Actual")
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    plot_comparison_svm(y_test, svm_preds)

elif model_choice == "KNN":
    st.subheader("พื้นที่ ที่มีความเสี่ยงในการเกิดแผ่นดินไหวมากที่สุด (KNN)")
    st.dataframe(X_test[["depth_km", "longitude", "latitude", "sig", "gap", "KNN_Pred", "Actual_Mag"]].nlargest(top_n, "KNN_Pred"))
    
    def evaluate_model(name, y_true, y_pred):
        Error = mean_absolute_error(y_true, y_pred)
        Errorpercent = (Error / y_true.mean()) * 100

        Accuracy = r2_score(y_true, y_pred)
        Accuracypercent = Accuracy * 100

        st.subheader(f"ประสิทธิภาพในการทำนายของ {name}")
        st.write(f"ค่าความคลาดเคลื่อน: {Errorpercent:.2f} %")
        st.write(f"ค่าความแม่นยำ: {Accuracypercent:.2f} %")
    
    evaluate_model("KNN ", y_test, knn_preds)

    def plot_comparison_knn(y_true, knn_preds):
        plt.figure(figsize=(6, 6))
        
        plt.scatter(y_true, knn_preds, color='#87CEEB', alpha=0.6, label="KNN Predictions")
        plt.scatter(y_true, y_true, color='pink', alpha=0.6, label="Actual Magnitude")
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
        
        plt.xlabel("Actual Magnitude")
        plt.ylabel("KNN Predicted Magnitude")
        plt.title("KNN Prediction vs Actual")
        plt.legend()

        plt.tight_layout()
        st.pyplot(plt)

    plot_comparison_knn(y_test, knn_preds)
