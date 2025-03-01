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
st.markdown("เป็นข้อมูลภายในเว็บไซต์ Kaggle ให้ข้อมูลเกี่ยวกับเหตุการณ์แผ่นดินไหวทั่วโลกอย่างครบถ้วน รวมถึงคุณลักษณะทางภูมิศาสตร์และการสั่นสะเทือนที่สำคัญ เช่น ขนาดแผ่นดินไหว ความลึก ตำแหน่ง และความสำคัญ ซึ่งทำให้เป็นแหล่งข้อมูลที่มีคุณค่าสำหรับนักวิจัย นักวิทยาศาสตร์ข้อมูล และนักธรณีวิทยาที่ศึกษารูปแบบและแนวโน้มของแผ่นดินไหว [Kaggle](https://www.kaggle.com/datasets/jay11rathod/earthquakes-dataset?resource=download)")
st.subheader("Data Features มีอะไรบ้าง ?")
st.markdown("**place**: ตำแหน่งทางภูมิศาสตร์ที่เกิดแผ่นดินไหว")
st.markdown("**mag**: ขนาดของแผ่นดินไหว ซึ่งใช้วัดการปลดปล่อยพลังงานของแผ่นดินไหว")
st.markdown("**magType**: ประเภทของการวัดขนาดแผ่นดินไหว เช่น ML (ขนาดท้องถิ่น), MD (ขนาดระยะเวลา), MB (ขนาดคลื่นร่างกาย) เป็นต้น")
st.markdown("**type**: ประเภทของเหตุการณ์ทางแผ่นดินไหว เช่น แผ่นดินไหว, การระเบิด, หรือเหตุการณ์ทางแผ่นดินไหวอื่นๆ")
st.markdown("**time**: วันที่และเวลาที่เกิดแผ่นดินไหวในรูปแบบที่อ่านได้")
st.markdown("**longitude & latitude**: พิกัดที่ระบุที่ตั้งที่แม่นยำของเหตุการณ์")
st.markdown("**depth_km**: ความลึกของแผ่นดินไหวในหน่วยกิโลเมตร")
st.markdown("**sig**: มาตรการของความสำคัญของแผ่นดินไหวที่รวมขนาดและผลกระทบ")
st.markdown("**net**: รหัสของเครือข่ายแผ่นดินไหวที่บันทึกเหตุการณ์")
st.markdown("**nst**: จำนวนสถานีแผ่นดินไหวที่ใช้ในการวิเคราะห์แผ่นดินไหว")
st.markdown("**dmin**: ระยะทางขั้นต่ำจากแผ่นดินไหวถึงสถานีแผ่นดินไหวที่ใกล้ที่สุด")
st.markdown("**rms**: ค่ารากที่สองเฉลี่ยของค่าความคลาดเคลื่อนของแอมพลิจูด ที่ใช้บ่งชี้ความแม่นยำของการวัด")
st.markdown("**gap**: ช่องว่างที่ใหญ่ที่สุดระหว่างสถานีแผ่นดินไหวที่วัดเหตุการณ์นี้ ซึ่งอาจส่งผลต่อความแม่นยำในการหาตำแหน่งแผ่นดินไหว")

st.subheader("โดยผมได้ทำการเลือกใช้ SVM และ KNN โดยมีขั้นตอนดังนี้")

st.subheader(" - SVM (Support Vector Machine)")
st.text("ขั้นตอนแรก ผมจะทำการนำเข้าข้อมูล ของแผ่นดินไหวเข้ามา พร้อมกับกำหนด feature ที่ต้องการใช้งาน เพื่อที่จะหาค่าของเป้าหมาย และ กำหนดค่า test กับ ค่า train")
codesvm1 = '''
df = pd.read_csv("dataset/Earthquakesdataset.csv")

features = ["depth_km", "longitude", "latitude", "sig", "gap"]
target = "mag"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

st.code(codesvm1, language='python')
st.text("ขั้นตอนที่สอง สร้างโมเดล SVM และ ทำการฝึกด้วยข้อมูลที่เตรียมไว้")

codesvm2 = '''
svm_model = SVR()
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
'''

st.code(codesvm2, language='python')

st.text("ขั้นตอนถัดไป คำนวณค่าความแม่นยำและค่าความคลาดเคลื่อนของโมเดล และ ทำการแสดงผลออกมาในรูปแบบ ของตาราง และ ความคลาดเคลื่อน")

codesvm3 = '''
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
'''
st.code(codesvm3, language='python')

st.text("ขั้นตอนสุดท้ายนำค่าของ Magnitude ที่ทำนายออกมา นำไปเปรียนบเที่ยบกับค่า Magnitude จริง โดยใช้กราฟแสดงผล")

codesvm4 = '''
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
'''
st.code(codesvm4, language='python')

st.subheader(" - KNN (K-Nearest Neighbors)")
st.text("ขั้นตอนแรก ผมจะทำการนำเข้าข้อมูล ของแผ่นดินไหวเข้ามา พร้อมกับกำหนด feature ที่ต้องการใช้งาน เพื่อที่จะหาค่าของเป้าหมาย และ กำหนดค่า test กับ ค่า train")
codeknn1 = '''
df = pd.read_csv("dataset/Earthquakesdataset.csv")

features = ["depth_km", "longitude", "latitude", "sig", "gap"]
target = "mag"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
'''

st.code(codeknn1, language='python')

st.text("ขั้นตอนที่สอง สร้างและฝึกโมเดล KNN โดยใช้ K=5 และ ทำการฝึกด้วยข้อมูลที่เตรียมไว้")

codeknn2 = '''
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
'''

st.code(codeknn2, language='python')

st.text("ขั้นตอนถัดไป คำนวณค่าความแม่นยำและค่าความคลาดเคลื่อนของโมเดล และ ทำการแสดงผลออกมาในรูปแบบ ของตาราง และ ความคลาดเคลื่อน")

codeknn3 = '''
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
'''
st.code(codeknn3, language='python')

st.text("ขั้นตอนสุดท้ายนำค่าของ Magnitude ที่ทำนายออกมา นำไปเปรียนบเที่ยบกับค่า Magnitude จริง โดยใช้กราฟแสดงผล")

codesvm4 = '''
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
'''
st.code(codesvm4, language='python')


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