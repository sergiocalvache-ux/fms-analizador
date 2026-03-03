import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. Configuración Médica de la App
st.set_page_config(page_title="FMS Analizador Pro", layout="centered")
st.title("🩺 Evaluación FMS: Deep Squat")
st.write("Posiciona al paciente de perfil y captura el punto de máxima flexión.")

# 2. Cargar el motor de IA
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# 3. Interfaz de Captura
img_file_buffer = st.camera_input("Capturar Movimiento")

if img_file_buffer is not None:
    # Convertir imagen para procesar
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    
    # Inferencia con YOLO
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    
    # Lógica de Ángulos (Copia de lo que hicimos en Colab)
    kp = results[0].keypoints.xy[0].cpu().numpy()
    if len(kp) > 15:
        # Hombro (5), Cadera (11), Rodilla (13), Tobillo (15)
        def get_angle(p1, p2):
            return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))
        
        ang_torso = get_angle(kp[5], kp[11])
        ang_tibia = get_angle(kp[13], kp[15])
        dif = abs(ang_torso - ang_tibia)

        # Mostrar Resultados Clínicos
        st.image(annotated_frame, caption="Análisis Biomecánico")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Ángulo Torso", f"{ang_torso:.1f}°")
        col2.metric("Ángulo Tibia", f"{ang_tibia:.1f}°")
        
        if dif < 10:
            st.success(f"SCORE FMS: 3 - Óptimo (Diferencia: {dif:.1f}°)")
        elif dif < 25:
            st.warning(f"SCORE FMS: 2 - Compensación (Diferencia: {dif:.1f}°)")
        else:
            st.error(f"SCORE FMS: 1 - Red Flag (Diferencia: {dif:.1f}°)")