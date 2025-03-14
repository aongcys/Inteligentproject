import streamlit as st
import pathlib


def load_css(file_name: str) -> None:
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


css_path = pathlib.Path("css/style.css")
if css_path.exists():
    load_css(css_path)
else:
    st.error(f"CSS file not found: {css_path.resolve()}")


st.title("🌍 The Earthquake")
st.subheader("The Earthquakes คือ โปรแกรมอะไร ?")
st.text("เป็นโปรแกรมที่จะวัดขนาดของของการเกิดแผ่นดินไหว และ พื้นที่ ที่มีความเสี่ยงของการเกิดแผ่นดินไหวมากที่สุด แล้วนำมาจัดแสดงเป็นข้อมูล เพื่อผลประโยชน์ต่างๆ เช่นการ ตรวจสอบพื้นที่ที่ได้รับผลกระทบ หรือ อื่นๆ")
st.image("https://i.pinimg.com/736x/b5/c0/b6/b5c0b60e0339a65fd3732725c5b8ed54.jpg")
st.subheader("ที่มาของ Dataset ?")
st.markdown(
    "เป็นข้อมูลภายในเว็บไซต์ Kaggle ให้ข้อมูลเกี่ยวกับเหตุการณ์แผ่นดินไหวทั่วโลกอย่างครบถ้วน รวมถึงคุณลักษณะทางภูมิศาสตร์และการสั่นสะเทือนที่สำคัญ เช่น ขนาดแผ่นดินไหว ความลึก ตำแหน่ง และความสำคัญ ซึ่งทำให้เป็นแหล่งข้อมูลที่มีคุณค่าสำหรับนักวิจัย นักวิทยาศาสตร์ข้อมูล และนักธรณีวิทยาที่ศึกษารูปแบบและแนวโน้มของแผ่นดินไหว [Kaggle](https://www.kaggle.com/datasets/jay11rathod/earthquakes-dataset?resource=download)")
st.subheader("Data Features มีอะไรบ้าง ?")
st.markdown("**place**: ตำแหน่งทางภูมิศาสตร์ที่เกิดแผ่นดินไหว")
st.markdown(
    "**mag**: ขนาดของแผ่นดินไหว ซึ่งใช้วัดการปลดปล่อยพลังงานของแผ่นดินไหว")
st.markdown("**magType**: ประเภทของการวัดขนาดแผ่นดินไหว เช่น ML (ขนาดท้องถิ่น), MD (ขนาดระยะเวลา), MB (ขนาดคลื่นร่างกาย) เป็นต้น")
st.markdown("**type**: ประเภทของเหตุการณ์ทางแผ่นดินไหว เช่น แผ่นดินไหว, การระเบิด, หรือเหตุการณ์ทางแผ่นดินไหวอื่นๆ")
st.markdown("**time**: วันที่และเวลาที่เกิดแผ่นดินไหวในรูปแบบที่อ่านได้")
st.markdown("**longitude & latitude**: พิกัดที่ระบุที่ตั้งที่แม่นยำของเหตุการณ์")
st.markdown("**depth_km**: ความลึกของแผ่นดินไหวในหน่วยกิโลเมตร")
st.markdown("**sig**: มาตรการของความสำคัญของแผ่นดินไหวที่รวมขนาดและผลกระทบ")
st.markdown("**net**: รหัสของเครือข่ายแผ่นดินไหวที่บันทึกเหตุการณ์")
st.markdown("**nst**: จำนวนสถานีแผ่นดินไหวที่ใช้ในการวิเคราะห์แผ่นดินไหว")
st.markdown(
    "**dmin**: ระยะทางขั้นต่ำจากแผ่นดินไหวถึงสถานีแผ่นดินไหวที่ใกล้ที่สุด")
st.markdown(
    "**rms**: ค่ารากที่สองเฉลี่ยของค่าความคลาดเคลื่อนของแอมพลิจูด ที่ใช้บ่งชี้ความแม่นยำของการวัด")
st.markdown("**gap**: ช่องว่างที่ใหญ่ที่สุดระหว่างสถานีแผ่นดินไหวที่วัดเหตุการณ์นี้ ซึ่งอาจส่งผลต่อความแม่นยำในการหาตำแหน่งแผ่นดินไหว")

st.subheader(
    "โดยผมได้ทำการเลือกใช้ SVM (Support Vector Machine) และ KNN (K-Nearest Neighbors) โดยมีขั้นตอนดังนี้")

st.text("ขั้นตอนแรก ผมจะทำการนำเข้าข้อมูล ของแผ่นดินไหวเข้ามา พร้อมกับกำหนด feature ที่ต้องการใช้งาน และ ตรวจสอบ พบทั้งค่า ว่าง หรือ ค่าที่หายไป และ ค่า outlier ผมจึงได้ทำการ cleansing data โดยค่าที่หายไป ผมจะแทนด้วย 0 และ แก้ outlier ด้วยการ IQR ")

code1 = '''
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
'''

st.code(code1, language='python')
st.text("ขั้นตอนที่สอง เป็นการนำข้อมูลที่ได้ทำการแก้ไขแล้ว มาเตรียมเพื่อรอการ เทรนโมเดลอีกทีนึง")

code2 = '''
features_model = ["depth_km", "longitude", "latitude", "sig", "gap"]
target = "mag"

X = df_cleaned[features_model]
y = df_cleaned[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
'''

st.code(code2, language='python')

st.text("ขั้นตอนถัดไป คือการนำข้อมูล หรือ feature ที่จะนำมาใช้สำหรับการ train มาทำการเทรนทั้งแบบ SVM และ KNN และ เพิ่ม dataframe ที่รวม features ผลทำนายจากทั้งสองโมเดล และค่าจริงที่ได้ออกมา")

code3 = '''
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
'''
st.code(code3, language='python')

st.text("ขั้นตอนถัดมา สร้าง dropdown ที่ให้ทำการเลือกโมเดลมาแสดง กับ slider เพื่อให้ผู้ใช้สามารถเลือกได้ว่าอยากได้พื้นที่ ที่มีความเสี่ยงกี่ที่ โดยการแสดงจะเป็นการเรียงตามค่าวามเสี่ยงที่จะเกิด")

code4 = '''
model_choice = st.selectbox("โมเดลที่ต้องการ", ("SVM", "KNN"))

top_n = st.slider(
    "เลือกว่าจะดูพื้นที่ ที่มีความเสี่ยงมากกี่พื้นที่:", 1, 30, 10)
'''
st.code(code4, language='python')

st.text("ในส่วนนี้ จะเป็นการ แสดงผลตัวของ SVM โดยจพมีการแสดง ตรางที่จัดอันดับความเสี่ยงมาแล้วม , ค่าความคลาดเคลื่อน , ค่าความแม่นยำ และ กราฟแสดงค่า mag ตัวโมลหาได้ เทียบกับ ค่าจริง")
code5 = '''
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
'''

st.code(code5, language='python')

st.text("แสดงผลตัวของ KNN โดยจพมีการแสดง ตรางที่จัดอันดับความเสี่ยงมาแล้วม , ค่าความคลาดเคลื่อน , ค่าความแม่นยำ และ กราฟแสดงค่า mag ตัวโมลหาได้ เทียบกับ ค่าจริง")

code6 = '''
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
'''

st.code(code6, language='python')

st.markdown(
    """
    <a href="/Machine_Learning_Model" style="text-decoration: none;">
        <div class="mainmachine">
            <h1 style="text-shadow: 2px 2px 5px black;">Go to the model</h1>
            <h3 style="text-shadow: 2px 2px 5px black; text-align: right;">Click me!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)
