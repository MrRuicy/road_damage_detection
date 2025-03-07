import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import time
import torch

# 检测是否有可用的GPU，并设置设备
gpu_available = torch.cuda.is_available()
device = "cuda" if gpu_available else "cpu"
st.sidebar.info(f"当前使用设备：{'GPU (CUDA)' if gpu_available else 'CPU'}")


# 缓存模型加载，避免每次重新加载
@st.cache_resource(show_spinner=False)
def load_model(model_path, device):
    try:
        # 加载模型
        model = YOLO(model_path)
        # 将模型移动到指定设备上
        model.model.to(device)
        return model
    except Exception as e:
        st.error(f"无法加载模型: {e}")
        st.stop()


# 加载YOLO模型，确保模型路径正确
model = load_model("./best.pt", device)

# 页面设置
st.title("道路病害检测系统")
st.sidebar.header("检测设置")
confidence_threshold = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.5, 0.01)

# 初始化 session_state，避免重复处理
if "processed_video_path" not in st.session_state:
    st.session_state.processed_video_path = None
if "video_processed" not in st.session_state:
    st.session_state.video_processed = False

# 文件上传组件
uploaded_file = st.file_uploader(
    "上传图片或视频文件",
    type=["jpg", "jpeg", "png", "mp4"],
    help="支持格式：JPG/PNG/MP4",
)


def process_frame(frame):
    """处理单帧并返回带标注的图像"""
    results = model(frame, conf=confidence_threshold)
    return results[0].plot()  # 返回带标注的BGR格式图像


def process_image(uploaded_file):
    """处理图片文件"""
    try:
        # 读取图片
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # 执行推理
        result_image = process_frame(image_np)
        result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        # 显示结果
        st.image(result_image_rgb, caption="检测结果", use_container_width=True)

        # 生成下载按钮
        img_bytes = cv2.imencode(".jpg", result_image)[1].tobytes()
        st.download_button(
            label="下载检测结果",
            data=img_bytes,
            file_name="detection_result.jpg",
            mime="image/jpeg",
        )

    except Exception as e:
        st.error(f"图片处理失败: {str(e)}")


def process_video_unified(uploaded_file):
    """实时显示视频处理，同时保存生成视频供下载"""
    try:
        # 如果视频已经处理过，直接显示下载按钮
        if st.session_state.video_processed:
            with open(st.session_state.processed_video_path, "rb") as f:
                video_bytes = f.read()
                st.download_button(
                    label="下载检测视频",
                    data=video_bytes,
                    file_name="detection_result.mp4",
                    mime="video/mp4",
                )
            return

        # 保存上传文件到临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建输出视频临时文件
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        progress_bar = st.progress(0)
        frame_placeholder = st.empty()  # 用于实时显示视频帧
        frame_count = 0

        with st.spinner("视频处理中，请稍候..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 对当前帧进行处理
                processed_frame = process_frame(frame)
                # 写入处理后的帧到输出视频
                out.write(processed_frame)
                # 转换颜色格式以适应 Streamlit 显示
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # 实时更新显示
                frame_placeholder.image(
                    processed_frame_rgb, channels="RGB", use_container_width=True
                )

                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
                # 根据处理速度可适当延时
                time.sleep(0.02)

        cap.release()
        out.release()
        os.unlink(video_path)  # 删除上传的临时文件

        # 缓存处理后的视频路径
        st.session_state.processed_video_path = output_path
        st.session_state.video_processed = True

        # 生成下载按钮
        with open(output_path, "rb") as f:
            video_bytes = f.read()
            st.download_button(
                label="下载检测视频",
                data=video_bytes,
                file_name="detection_result.mp4",
                mime="video/mp4",
            )

    except Exception as e:
        st.error(f"视频处理失败: {str(e)}")


# 主处理逻辑
if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[0]

    if file_type == "image":
        process_image(uploaded_file)
    elif file_type == "video":
        process_video_unified(uploaded_file)
    else:
        st.error("不支持的文件类型")

# 使用说明
st.markdown(
    """
### 使用说明
1. 上传道路图片或视频（支持JPG/PNG/MP4格式）。
2. 调整侧边栏的置信度阈值（默认0.5）。
3. 对于视频：
   - 系统会优先使用GPU（如果可用）进行处理，并实时显示检测结果。
   - 处理完成后会生成下载按钮，**不会重复处理**。
4. 查看检测结果并下载处理后的文件。

**注意**：视频处理可能需要较长时间，请耐心等待进度条完成；实时显示模式下，视频帧处理速度受硬件和网络环境影响。
"""
)
