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

st.title("The Intelligent project")
st.text("โดยผมได้ทำ การเลือก สอง dataset มาเพื่อตรวจสอบและพัฒนา โดยจะมีการพัฒนาโมเดลโดยจะมี สองประเภทคือ Machine Learning และ Neural Network ")
st.subheader("- Machine Learning")

st.markdown(
    """
    <a href="/Machine_Learning" style="text-decoration: none;">
        <div class="mainmachine">
            <h1 style="text-shadow: 2px 2px 5px black;">The Earthquakes</h1>
            <h3 style="text-shadow: 2px 2px 5px black; text-align: right;">Click me!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)

st.subheader("- Neural Network")

st.markdown(
    """
    <a href="/Neural_Network" style="text-decoration: none;">
        <div class="mainnaural">
            <h1 style="text-shadow: 2px 2px 5px black;">Coffee Pediction</h1>
            <h3 style="text-shadow: 2px 2px 5px black; text-align: right;">Click me!</h3>
        </div>
    </a>
    """,
    unsafe_allow_html=True
)
