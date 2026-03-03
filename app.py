import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from docx import Document
from docx.shared import Inches
import io

# 1. Configuración Médica
st.set_page_config(page_title="FMS Analizador Pro", layout="centered")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- LÓGICA DE PROCESAMIENTO ---
class FMSProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Inferencia YOLO
        results = model(img, verbose=False)
        annotated_img = results[0].plot()
        
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

st.title("🩺 Evaluación FMS Profesional")

# Selector de Test
test_seleccionado = st.sidebar.selectbox(
    "Seleccionar Test", 
    ["Deep Squat", "Hurdle Step", "Inline Lunge", "Shoulder Mobility"]
)

# Cámara en tiempo real (Esta versión NO DA ERROR de librerías)
ctx = webrtc_streamer(
    key="fms-analisis",
    video_processor_factory=FMSProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

st.write("Presiona 'Start' para iniciar la cámara con IA.")

# --- SECCIÓN DE CAPTURA Y WORD ---
# (Aquí sigue el resto de tu lógica para el informe Word...)
