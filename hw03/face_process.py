import face_recognition
import cv2
import numpy as np
import os

def detect_faces(image):
    """
    人脸检测：返回人脸位置框 (top, right, bottom, left)
    输入：cv2读取的BGR格式图片
    输出：人脸位置列表，每个元素为(top, right, bottom, left)
    """
    # 转换为RGB格式（face_recognition要求RGB输入）
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 使用HOG模型检测人脸（CPU友好，适合Windows）
    face_locations = face_recognition.face_locations(rgb_image, model="hog")
    return face_locations

def encode_faces(image, face_locations=None):
    """
    人脸特征编码：生成128维特征向量
    输入：cv2读取的BGR格式图片、可选的人脸位置框
    输出：人脸编码列表、人脸位置列表
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 若未传入位置框，先检测人脸
    if face_locations is None:
        face_locations = face_recognition.face_locations(rgb_image)
    # 生成128维特征编码
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    return face_encodings, face_locations

def load_known_faces(known_faces_dir="known_faces"):
    """
    加载已知人脸库：从目录读取图片，生成对应编码和姓名
    输入：已知人脸库目录路径
    输出：已知人脸编码列表、对应姓名列表
    """
    known_encodings = []
    known_names = []
    # 遍历目录下所有图片文件
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            # 文件名作为姓名（如zhangsan.jpg → 姓名zhangsan）
            name = os.path.splitext(filename)[0]
            # 加载图片并生成编码
            image = face_recognition.load_image_file(os.path.join(known_faces_dir, filename))
            # 取第一张人脸的编码（假设每张图只有1个目标人脸）
            encoding = face_recognition.face_encodings(image)[0]
            known_encodings.append(encoding)
            known_names.append(name)
    return known_encodings, known_names

def compare_faces(unknown_encoding, known_encodings, known_names, tolerance=0.6):
    """
    比对未知人脸与已知库，返回匹配结果
    输入：未知人脸编码、已知编码列表、已知姓名列表、匹配阈值（默认0.6，越小越严格）
    输出：匹配姓名、匹配距离（越小越相似）
    """
    # 比对人脸，返回布尔列表
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding, tolerance=tolerance)
    # 计算人脸距离（相似度）
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    # 找到最相似的人脸
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return known_names[best_match_index], face_distances[best_match_index]
    # 无匹配则返回Unknown
    return "Unknown", face_distances[best_match_index]

def draw_faces(image, face_locations, names=None):
    """
    在图片上绘制人脸框和识别结果
    输入：原始图片、人脸位置列表、识别结果列表（可选）
    输出：绘制后的图片
    """
    img_copy = image.copy()
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # 绘制绿色人脸框
        cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 255, 0), 2)
        # 绘制识别结果标签
        if names and i < len(names):
            name, distance = names[i]
            label = f"{name} ({1 - distance:.2f})"  # 显示姓名+置信度（1-距离）
            # 绘制标签背景框
            cv2.rectangle(img_copy, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            # 绘制白色文字
            cv2.putText(img_copy, label, (left + 6, bottom - 6), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return img_copy
