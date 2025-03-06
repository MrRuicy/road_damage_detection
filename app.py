import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 加载 YOLOv11 模型
model = YOLO("./best.pt")

# 设置页面标题
st.title("YOLOv11 道路病害检测")

# 侧边栏设置
st.sidebar.header("设置")
confidence_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)
show_labels = st.sidebar.checkbox("显示标签", value=True)
show_conf = st.sidebar.checkbox("显示置信度", value=True)
save_results = st.sidebar.checkbox("保存推理结果", value=True)
save_folder = st.sidebar.text_input("保存路径", value="results")

# 创建保存文件夹
if save_results and not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 上传图片或视频
uploaded_file = st.file_uploader(
    "上传图片或视频", type=["jpg", "png", "jpeg", "mp4", "avi"]
)

# 摄像头检测
use_camera = st.sidebar.checkbox("启用摄像头检测")


def process_frame(frame, confidence_threshold, show_labels, show_conf):
    """处理帧并进行推理"""
    results = model(frame, conf=confidence_threshold)
    for result in results:
        frame = result.plot(
            labels=show_labels,  # 是否显示标签
            conf=show_conf,  # 是否显示置信度
            line_width=2,  # 边界框线宽
        )
    return frame


if use_camera:
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("无法打开摄像头")
        st.stop()

    # 设置视频保存路径
    if save_results:
        output_video_path = os.path.join(save_folder, "camera_detection.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(
            output_video_path, fourcc, fps, (frame_width, frame_height)
        )

    # 显示摄像头推理结果
    stframe = st.empty()
    stop_button = st.button("停止摄像头检测")

    if "stop_camera" not in st.session_state:
        st.session_state.stop_camera = False

    while cap.isOpened() and not st.session_state.stop_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("无法读取摄像头帧")
            break

        # 对帧进行推理
        frame = process_frame(frame, confidence_threshold, show_labels, show_conf)

        # 显示帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB", use_container_width=True)

        # 保存帧
        if save_results:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # 检查是否按下停止按钮
        if stop_button:
            st.session_state.stop_camera = True

    # 释放资源
    cap.release()
    if save_results:
        out.release()
        st.success(f"摄像头检测结果已保存到: {output_video_path}")

elif uploaded_file is not None:
    # 判断文件类型
    file_type = uploaded_file.type.split("/")[0]  # 获取文件类型（image 或 video）

    if file_type == "image":
        # 处理图片
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 对图片进行推理
        img_with_boxes = process_frame(
            img, confidence_threshold, show_labels, show_conf
        )

        # 将 OpenCV 图片转换为 PIL 图片以便在 Streamlit 中显示
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img_with_boxes = Image.fromarray(img_with_boxes)

        # 显示推理结果
        st.image(img_with_boxes, caption="检测结果", use_container_width=True)

        # 保存推理结果
        if save_results:
            save_path = os.path.join(save_folder, "detection_result.jpg")
            img_with_boxes.save(save_path)
            st.success(f"检测结果已保存到: {save_path}")

    elif file_type == "video":
        # 处理视频
        video_bytes = uploaded_file.read()
        video_path = os.path.join(save_folder, "uploaded_video.mp4")
        with open(video_path, "wb") as f:
            f.write(video_bytes)

        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("无法打开视频文件")
            st.stop()

        # 获取视频信息
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # 设置视频保存路径
        if save_results:
            output_video_path = os.path.join(save_folder, "detection_result.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_video_path, fourcc, fps, (frame_width, frame_height)
            )

        # 显示视频推理结果
        stframe = st.empty()
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 对帧进行推理
            frame = process_frame(frame, confidence_threshold, show_labels, show_conf)

            # 显示帧
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_container_width=True)

            # 保存帧
            if save_results:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # 更新进度条
            current_frame += 1
            progress_bar.progress(current_frame / total_frames)

        # 释放资源
        cap.release()
        if save_results:
            out.release()
            st.success(f"检测结果已保存到: {output_video_path}")
