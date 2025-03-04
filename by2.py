import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.feature import local_binary_pattern

# ตั้งค่าฟอนต์ภาษาไทยใน Matplotlib
plt.rcParams['font.family'] = 'Arial Unicode MS'

# ตั้งค่าหน้าตาเว็บให้รองรับสมาร์ทโฟน และพื้นหลังสีเขียวเข้ม
st.set_page_config(page_title="วิเคราะห์โอกาสปนเปื้อน BY2 ในทุเรียน", layout="wide")
st.markdown("""
    <style>
        .stApp { background-color: #EFF8EFFF; }
        h1, h2, h3 { color: #ffcc00; text-align: center; }
        .note-text { color: #5826BBFF; font-size: 22px; font-weight: bold; text-align: center; }
        .guide-text { color: #ffcc00; font-size: 20px; font-weight: bold; text-align: left; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>🔍 ระบบวิเคราะห์โอกาสปนเปื้อน BY2 ในทุเรียน</h1>", unsafe_allow_html=True)

# ฟังก์ชันวิเคราะห์ภาพ
def analyze_durian(image):
    img = np.array(image.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_hue = np.mean(hsv[:, :, 0])  
    avg_saturation = np.mean(hsv[:, :, 1])  

    color_risk = 1 if (20 < avg_hue < 35 and avg_saturation > 120) else 0
    color_warning = "พบสีเหลืองผิดปกติ!" if color_risk else "สีปกติ"

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method='uniform')
    texture_score = np.histogram(lbp, bins=10)[0].var()

    texture_risk = 1 if texture_score < 500 else 0
    texture_warning = "พบพื้นผิวเรียบผิดปกติ!" if texture_risk else "พื้นผิวปกติ"

    contamination_risk = (color_risk + texture_risk) / 2 * 100

    return color_warning, texture_warning, contamination_risk, hsv, lbp

# UI ส่วนอัปโหลดภาพ
st.markdown("<h3>📤 อัปโหลดภาพทุเรียน</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 ภาพต้นฉบับ", use_column_width=True)

    color_warning, texture_warning, contamination_risk, hsv, lbp = analyze_durian(image)
    st.markdown("### 🔬 ผลการวิเคราะห์", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background-color:#ff4b4b; padding:10px; border-radius:10px; text-align:center;">
                <p style="font-size:22px; color:white; font-weight:bold;">🚨 สีของเปลือกและหนาม :  {color_warning}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color:#ff4b4b; padding:10px; border-radius:10px; text-align:center;">
                <p style="font-size:22px; color:white; font-weight:bold;">🔍 ลักษณะของสีพื้นผิว  :   {texture_warning}</p>
            </div>
        """, unsafe_allow_html=True)
    
    # แสดงระดับความเสี่ยง
    st.markdown("### 🚨 ระดับความเสี่ยงการปนเปื้อน", unsafe_allow_html=True)
    st.progress(int(contamination_risk))
    
    risk_color = "#00cc00" if contamination_risk <= 30 else "#F5EDD0FF" if contamination_risk <= 70 else "#ff0000"
    risk_text = f"<h2 style='color:{risk_color}; text-align:center;'>🔴 ความเสี่ยงสูง  :   {contamination_risk:.2f}%</h2>" if contamination_risk > 70 else \
                f"<h2 style='color:{risk_color}; text-align:center;'>🟡 ความเสี่ยงปานกลาง  :   {contamination_risk:.2f}%</h2>" if contamination_risk > 30 else \
                f"<h2 style='color:{risk_color}; text-align:center;'>🟢 ความเสี่ยงต่ำ  :   {contamination_risk:.2f}%</h2>"
    st.markdown(risk_text, unsafe_allow_html=True)
    
    # แสดงภาพวิเคราะห์ (ลดขนาดภาพลง)
    st.markdown("### 🖼️ ภาพวิเคราะห์", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    
    axes[0].imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
    axes[0].set_title("การวิเคราะห์สี (HSV)", fontsize=10, color='#111FEDFF')
    axes[0].axis("off")

    axes[1].imshow(lbp, cmap="gray")
    axes[1].set_title("การวิเคราะห์พื้นผิว (LBP)", fontsize=10, color='#4909EBFF')
    axes[1].axis("off")

    st.pyplot(fig)

# แสดงคำเตือนเรื่องความแม่นยำ
st.markdown("<h2 style='color:#ffcc00; font-size:24px;'>⚠️ หมายเหตุ:</h2>", unsafe_allow_html=True)
st.markdown("""
    <p style='font-size:16px; color:blue;'>
        การวิเคราะห์นี้เป็นการประเมินเบื้องต้น อาจมีความคลาดเคลื่อนขึ้นอยู่กับคุณภาพของภาพ <br>
        กรุณาใช้ควบคู่กับการตรวจสอบด้วยตาเปล่า
    </p>
    """, unsafe_allow_html=True)

# คำแนะนำการใช้งาน
st.markdown("<p class='guide-text'>ℹ️ คำแนะนำการใช้งาน</p>", unsafe_allow_html=True)
with st.expander("ดูรายละเอียด", expanded=False):
    st.markdown("""
    - 📏 **ระยะถ่ายภาพ**: ควรอยู่ห่างจากทุเรียนประมาณ **30-50 ซม.**  
    - 📸 **มุมกล้อง**: ถ่ายจากด้านบนตรงๆ หรือเอียงไม่เกิน **45 องศา**  
    - 💡 **แสง**: ใช้แสงธรรมชาติหรือแสงไฟสีขาว หลีกเลี่ยงแสงสีเหลืองหรือแสงน้อย  
    - 🔍 **ความคมชัด**: ภาพต้องไม่เบลอและต้องโฟกัสที่พื้นผิวของทุเรียน  
    """, unsafe_allow_html=True)
