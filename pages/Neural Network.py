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
st.markdown(
    "เป็นข้อมูลภายในเว็บไซต์ Kaggle ให้ข้อมูลเกี่ยวกับ ข้อมุลของเมล็ดกาแฟ อาราบิก้า โดยเป็นข้อมูลที่มาจาก CQI ใน พฤษภาคม ปี 2023 ที่มีฐานข้อมูลออนไลน์ที่เป็นแหล่งข้อมูลสำหรับผู้เชี่ยวชาญด้านกาแฟและผู้ที่สนใจศึกษาเกี่ยวกับคุณภาพและความยั่งยืนของกาแฟ ฐานข้อมูลนี้ประกอบไปด้วยข้อมูลหลากหลายด้าน เช่น การผลิตกาแฟ, การแปรรูป, และการประเมินรสชาติ นอกจากนี้ยังมีข้อมูลเกี่ยวกับพันธุกรรมของกาแฟ, ประเภทของดิน, และปัจจัยอื่น ๆ ที่อาจส่งผลต่อคุณภาพของกาแฟ [Kaggle](https://www.kaggle.com/datasets/fatihb/coffee-quality-data-cqi)")
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

st.text("ขั้นตอนแรก นำเข้าข้อมูลและตรวจสอบว่า มีจุดไหนปกพร้องไป แล้ว ทำการเลือกข้อมูลที่ต้องการนำมาใช้ กับ แก้ข้อมูลให้ถูกต้องเช่นการเติมค่า mean ลงไปในช่องที่มี่ค่า null หรือ Nan และแสดงตัวอย่างข้อมูล")

code1 = '''
df = pd.read_csv("dataset/arabicadata.csv")
st.subheader("ตัวอย่างของชุดข้อมูล")

df = df[["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
         "Uniformity", "Clean.Cup", "Sweetness", "Country.of.Origin"]]

numeric_cols = ["Aroma", "Flavor", "Aftertaste", "Acidity", "Body", "Balance",
                "Uniformity", "Clean.Cup", "Sweetness"]
imputer = SimpleImputer(strategy="mean")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

st.dataframe(df.head())
'''

st.code(code1, language='python')

st.text("ขั้นตอนถัดไป กำหนด colum ที่จะนำมาใช้ในการเทรน และ เก็บประเทศเอาไว้ ถัดมา ทำการ MinMaxScaler ที่จะทำให้ข้อมูลที่ ต้องการมาเทรนนนั้น มีค่าอยู่ในช่วง 0 - 1 และ นำประเทศมาแปลงเป็นตัวเลขเผื่อ สะดวกต่อการทดสอบ และ ทำนาย สุดท้ายคือ แสดงประเทศที่มีอยู่ในชุดข้อมูล")

code2 = '''
country_col = df["Country.of.Origin"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(df[numeric_cols])
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
scaled_df["Country.of.Origin"] = country_col.values

country = LabelEncoder()
country.fit(df["Country.of.Origin"])

st.subheader("ประเทศที่มีอยู่ในชุดข้อมูล")
st.write(np.array(country.classes_).reshape(1, -1))
'''
st.code(code2, language='python')

st.text("ในขั้นตอนนี้ผม เตรียมพร้อมข้อมูลที่จะทำการ เทรน โดย จะมีการ แยกค่าที่ต้องทำนาย และ ค่าที่ต้องการนำมา เทรน และ เติมค่าที่ขาดหายไปด้วยค่าเฉลี่ย และ ใช้ StandardScaler ทำให้ค่ามีการกระจายที่เหมาะสม")

code3 = '''
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
model = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                      alpha=0.001, max_iter=100, random_state=42)
model.fit(traindata, trantarget)
'''

st.code(code5, language='python')

st.text("ถัดไปคือการที่ สร้าง input เพื่อให้ผู้ใช้ใส่คะแนน")

code6 = '''
    inputs = {}
for col in fetcol:
    inputs[col] = st.number_input(f"{col}", value=5.0, step=0.1)
'''
st.code(code6, language='python')

st.text("ขั้นตอนสุดท้าย เมื่อได้ทำกาารใส่ input ที่ต้องการก็จะสามารถกดปุ่มเผื่อหา ว่าที่มาของเมล็ดกาแฟมาจากที่ไหน โดยจะเก็บค่า input ที่ได้มาเป็น array และหากผู้ใช้ไม่ได้ใส่ค่าในบาง input ก็จะเติมค่าที่ขาดหายไปด้วย SimpleImputer และ ปรับสเกลของ input ให้เหมือนกับชุดฝึก และ ตัวโมเดลจะทำการคำนวณ และ หาผลลัพท์")

code7 = '''
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
