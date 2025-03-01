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


st.title("Coffee Prediction")
st.subheader("Coffee Prediction คือ โปรแกรมอะไร ?")
st.text("เป็นโปรแกรมที่จะหาว่ากาแฟ arabica ตัวอย่างที่ได้กรอกผล มานั้นมีต้นกำเหนิดมาจากประเทศอะไร")
st.image("https://i.pinimg.com/736x/41/45/88/414588b8651bfe51acbf2feeeb333f7f.jpg", width=300)
st.subheader("ที่มาของ Dataset ?")
st.markdown("เป็นข้อมูลภายในเว็บไซต์ Kaggle ให้ข้อมูลเกี่ยวกับ ข้อมุลของเมล็ดกาแฟ อาราบิก้า โดยเป็นข้อมูลที่มาจาก CQI ใน พฤษภาคม ปี 2023 ที่มีฐานข้อมูลออนไลน์ที่เป็นแหล่งข้อมูลสำหรับผู้เชี่ยวชาญด้านกาแฟและผู้ที่สนใจศึกษาเกี่ยวกับคุณภาพและความยั่งยืนของกาแฟ ฐานข้อมูลนี้ประกอบไปด้วยข้อมูลหลากหลายด้าน เช่น การผลิตกาแฟ, การแปรรูป, และการประเมินรสชาติ นอกจากนี้ยังมีข้อมูลเกี่ยวกับพันธุกรรมของกาแฟ, ประเภทของดิน, และปัจจัยอื่น ๆ ที่อาจส่งผลต่อคุณภาพของกาแฟ [Kaggle](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)")
st.subheader("Data Features ที่นำมาใช้งานมีอะไรบ้าง ?")
st.markdown("**Aroma**: กลิ่นหอมของกาแฟ")
st.markdown("**Flavor**: รสชาติของกาแฟ เช่น หวาน, ขม, เปรี้ยว และกลิ่นรสอื่น ๆ")
st.markdown("**Aftertaste**: รสที่หลงเหลือหลังจากดื่มกาแฟ")
st.markdown("**Acidity**: ความเปรี้ยวสดใสของกาแฟ")
st.markdown("**Body**: เนื้อสัมผัสของกาแฟในปาก")
st.markdown("**Balance**: ความสมดุลขององค์ประกอบรสชาติต่าง ๆ")
st.markdown("**Uniformity**: ความสม่ำเสมอของรสชาติในแต่ละแก้ว")
st.markdown("**Clean Cup**: ความสะอาดของรสชาติ ไม่มีรสแปลกปลอม")
st.markdown("**Sweetness**: ความหวานที่เป็นธรรมชาติ เช่น คาราเมล, ผลไม้, ดอกไม้")
st.markdown("**Total Cup Points**: คะแนนรวมที่ได้จากคุณสมบัติด้านบน")

st.subheader("ขั้นตอนในการพัฒนาโมเดล มีอะไรบ้าง")

st.text("ขั้นตอนแรก นำเข้าข้อมูลและตรวจสอบว่า มีจุดไหนปกพร้องไป แล้ว ทำการเลือก้อมูลที่ต้องการนำมาใช้ กับ แก้ข้อมูลให้ถูกต้อง โดย df จะเก็บข้อมูลที่โหลดจากไฟล์ CSV และแสดงตัวอย่างข้อมูล โดย ใช้ st")

code1 = '''
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
'''

st.code(code1, language='python')

st.text("ขั้นตอนถัดไป แปลงข้อมูลประเทศที่เป็นหมวดหมู่ (Country.of.Origin) ให้เป็นตัวเลข และ แสดงประเทศที่มีอยู่ภายในชุดข้อมูล")

code2 = '''
  country = LabelEncoder()
  country.fit(df["Country.of.Origin"])

  st.subheader("ประเทศที่มีอยู่ในชุดข้อมูล")
  st.write(np.array(country.classes_).reshape(1, -1))
'''
st.code(code2, language='python')

st.text("ในขั้นตอนนี้ผม จะให้ผู้ใช้ได้ทำการกรอกข้อมูลของกาแฟ")

code3 = '''
inputs = {}
for col in df.columns[:-1]: 
    inputs[col] = st.number_input(f"{col}", value=5.0, step=0.1)
'''
st.code(code3, language='python')

st.text("ต่อมาผมจะทำให้ข้อมูลถุกแปลงเป็น ตัวเลขเพื่อที่จะสามารถนำไปใช้งานได้ และ แบ่ง ชุดข้อมูล ที่ใช้ในการเทรน กับ เอาไว้ทดสอบออกจากกัน และ ทำการเตรียมข้อมูลให้ สมบูรณ์ก่อนจะนำไป พัฒนา")
code4 = '''
    encoder = LabelEncoder()
    finedata = encoder.fit_transform(finedata)

    traindata, testdata,trantarget, testtarget = train_test_split(usedata, finedata, test_size=0.3, random_state=42)

    imputer = SimpleImputer(strategy="mean")
    traindata = imputer.fit_transform(traindata)
    testdata = imputer.transform(testdata)

    scaler = StandardScaler()
    traindata = scaler.fit_transform(traindata)
    testdata = scaler.transform(testdata)
'''

st.code(code4, language='python')

st.text("เมื่อผมได้ข้อมูลที่มีความสมบูรณ์ มาแล้ว จะทำการพัฒนาโมเดลโดยใช้ MLPClassifier ที่เป็นโมเดล Neural Network ในการฝึกจากข้อมูลที่ได้เตรียมไว้")

code5 = '''
  mlp_classifier = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                                  alpha=0.01, max_iter=1000, random_state=42)
  mlp_classifier.fit(traindata, trantarget)
'''

st.code(code5, language='python')

st.text("เมื่อได้ฝึกตังโมเดลแล้วก็จะนำมาใช้ ในการทำนายแหล่งที่มาของกาแฟจาก ข้อมูลที่ได้ทำการ input เข้ามานั้นเอง")

code6 = '''
    input_values = np.array([[inputs[col] for col in usedata.columns]])
    input_values = imputer.transform(input_values)
    input_values = scaler.transform(input_values)

    prediction_probs = mlp_classifier.predict_proba(input_values)[0]
    prediction = np.argmax(prediction_probs) # เลือกประเทศที่มีค่าความน่าจะเป็นมากสุด
    predicted_country = encoder.inverse_transform([prediction])[0] # แปลงค่าตัวเลขกลับเป็นชื่อประเทศ
    confidence = prediction_probs[prediction] * 100
'''
st.code(code6, language='python')

st.text("ขั้นตอนสุดท้ายคือการ แสดงผลลัพท์ที่ได้ออกมาเป็น ประเทศ และ ความมั่นใจในการทำนายนั้นเอง")

code7 = '''
  st.success(f"แหล่งที่มาของกาแฟคือ : {predicted_country}")
  st.success(f"ความมั่นใจในการทำนายคือ : {confidence:.2f} %")
'''
st.code(code7, language='python')

st.markdown(
    """
    <a href="/Neural_Network_Model" style="text-decoration: none;">
        <div class="mainnaural">
            <h1 style="text-shadow: 2px 2px 5px black;">Go to the model</h1>
            <h3 style="text-shadow: 2px 2px 5px black; text-align: right;">Click me!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)