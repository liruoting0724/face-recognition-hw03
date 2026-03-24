# hw03 人脸检测与识别系统
## 项目简介
本项目基于`face_recognition`库实现人脸检测、128维特征编码、人脸比对功能，通过`Streamlit`搭建Web界面，支持图片上传/示例图选择、人脸框选可视化、识别结果展示。

---
## 项目结构
hw03/
├── app.py # Streamlit Web 界面主程序
├── face_process.py # 人脸检测、编码、比对核心逻辑
├── requirements.txt # 项目依赖清单
├── README.md # 项目说明文档
├── examples/ # 示例图片目录（用于界面选择示例图）
└── known_faces/ # 已知人脸库目录（文件名 = 姓名，用于人脸识别）
