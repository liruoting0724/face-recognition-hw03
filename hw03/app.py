import streamlit as st
import cv2
import numpy as np
from face_process import detect_faces, encode_faces, load_known_faces, compare_faces, draw_faces
import os

# -------------------------- 页面基础配置 --------------------------
st.set_page_config(
    page_title="人脸检测与识别系统",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("🎯 人脸检测与识别系统（hw03）")
st.markdown("---")

# -------------------------- 加载已知人脸库（缓存优化） --------------------------
@st.cache_resource  # 缓存加载结果，避免重复加载
def load_known_data():
    # 检查known_faces目录是否存在，不存在则创建
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        return [], []
    return load_known_faces("known_faces")

known_encodings, known_names = load_known_data()

# -------------------------- 图片输入模块（上传/示例图） --------------------------
st.subheader("📸 图片输入")
input_method = st.radio(
    "选择图片输入方式",
    ["上传本地图片", "选择示例图片"],
    horizontal=True
)

# 初始化图片变量
image = None

# 方式1：上传本地图片
if input_method == "上传本地图片":
    uploaded_file = st.file_uploader(
        "请选择图片文件（支持jpg/png/jpeg）",
        type=["jpg", "jpeg", "png"],
        help="建议上传正面、清晰、无遮挡的人脸图片"
    )
    if uploaded_file is not None:
        # 读取上传的图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# 方式2：选择示例图片（examples目录）
else:
    # 检查examples目录是否存在，不存在则创建
    if not os.path.exists("examples"):
        os.makedirs("examples")
        st.warning("示例图片目录为空，请先在examples目录添加图片！")
    else:
        # 获取示例图片列表
        example_files = [f for f in os.listdir("examples") if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not example_files:
            st.warning("示例图片目录为空，请先在examples目录添加图片！")
        else:
            selected_example = st.selectbox("请选择示例图片", example_files)
            # 读取示例图片
            image = cv2.imread(os.path.join("examples", selected_example))

# -------------------------- 人脸检测与识别（核心逻辑） --------------------------
if image is not None:
    # 分栏展示原始图和结果图
    col1, col2 = st.columns(2)
    
    # 左侧：原始图片
    with col1:
        st.subheader("原始图片")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)
    
    # 右侧：检测结果
    with st.spinner("🔍 正在检测人脸，请稍候..."):
        # 1. 执行人脸检测
        face_locations = detect_faces(image)
        
        if not face_locations:
            st.warning("⚠️ 未检测到人脸，请更换图片！")
        else:
            # 2. 生成人脸特征编码
            face_encodings, _ = encode_faces(image, face_locations)
            
            # 3. 与人脸库比对（若有人脸库）
            results = []
            if known_encodings:
                for encoding in face_encodings:
                    name, distance = compare_faces(encoding, known_encodings, known_names)
                    results.append((name, distance))
            else:
                # 无人脸库时，仅显示检测框
                results = [("Detected", 0.0) for _ in face_locations]
            
            # 4. 绘制结果图片
            result_image = draw_faces(image, face_locations, results)
            
            # 展示结果图
            with col2:
                st.subheader("检测/识别结果")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # 5. 展示详细识别信息
            st.markdown("---")
            st.subheader("📊 识别详情")
            for i, (name, distance) in enumerate(results):
                st.write(f"✅ 人脸 {i+1}：识别结果 = **{name}**，置信度 = **{1 - distance:.2f}**")

# -------------------------- 侧边栏说明 --------------------------
with st.sidebar:
    st.subheader("📖 使用说明")
    st.markdown("""
    1.  **图片输入**：支持上传本地图片或选择示例图片
    2.  **人脸检测**：自动检测图片中的人脸，用绿色框标注
    3.  **人脸识别**：与`known_faces`目录中的人脸库比对，显示姓名和置信度
    4.  **置信度说明**：数值越接近1，识别准确率越高
    """)
    st.markdown("---")
    st.subheader("⚙️ 环境依赖")
    st.code("""
    face-recognition==1.3.0
    streamlit==1.32.2
    opencv-python==4.9.0.80
    numpy==1.26.4
    dlib==19.24.2
    """)
