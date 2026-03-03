import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. Configuración Médica de la App
st.set_page_config(page_title="FMS Analizador Pro", layout="centered")

# Sidebar para gestión de datos y selección de test
st.sidebar.title("Configuración Clínica")
nombre_paciente = st.sidebar.text_input("Nombre del Paciente", "Paciente Genérico")
test_seleccionado = st.sidebar.selectbox(
    "Seleccionar Test FMS",
    ["Deep Squat", "Hurdle Step", "Inline Lunge"]
)

st.title(f"🩺 Evaluación FMS: {test_seleccionado}")
st.write(f"Paciente: **{nombre_paciente}**")
st.write("Captura el punto de máxima exigencia biomecánica para el análisis.")

# 2. Cargar el motor de IA
@st.cache_resource
def load_model():
    # Usamos yolov8n-pose para máxima velocidad en servidor web
    return YOLO('yolov8n-pose.pt')

model = load_model()

# 3. Interfaz de Captura
img_file_buffer = st.camera_input("Capturar Movimiento")

# Función auxiliar para ángulos
def get_angle(p1, p2):
    return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    kp = results[0].keypoints.xy[0].cpu().numpy()

    if len(kp) > 15:
        st.image(annotated_frame, caption=f"Análisis Biomecánico - {test_seleccionado}")
        
        # --- LÓGICA DE DIAGNÓSTICO SEGÚN TEST ---
        
        if test_seleccionado == "Deep Squat":
            # Puntos: Hombro (5), Cadera (11), Rodilla (13), Tobillo (15)
            ang_torso = get_angle(kp[5], kp[11])
            ang_tibia = get_angle(kp[13], kp[15])
            dif = abs(ang_torso - ang_tibia)

            col1, col2 = st.columns(2)
            col1.metric("Ángulo Torso", f"{ang_torso:.1f}°")
            col2.metric("Ángulo Tibia", f"{ang_tibia:.1f}°")

            if dif < 10:
                st.success(f"SCORE FMS: 3 - Óptimo (Diferencia: {dif:.1f}°)")
            elif dif < 25:
                st.warning(f"SCORE FMS: 2 - Compensación (Diferencia: {dif:.1f}°)")
            else:
                st.error(f"SCORE FMS: 1 - Red Flag (Diferencia: {dif:.1f}°)")

        elif test_seleccionado == "Hurdle Step":
            # Puntos: Hombro (5/6), Cadera (11/12), Tobillo (15/16)
            # Analizamos la estabilidad del torso (verticalidad)
            verticalidad = get_angle(kp[5], kp[11]) # Alineación vertical del tronco
            
            st.metric("Inclinación Tronco", f"{verticalidad:.1f}°")
            
            if verticalidad < 5:
                st.success("SCORE FMS: 3 - Estabilidad lumbo-pélvica excelente")
            elif verticalidad < 15:
                st.warning("SCORE FMS: 2 - Compensación detectada en el plano sagital")
            else:
                st.error("SCORE FMS: 1 - Red Flag: Pérdida notable del control postural")

        elif test_seleccionado == "Inline Lunge":
            # Analizamos la alineación de la rodilla trasera (valgo dinámico)
            # Cadera (11), Rodilla (13), Tobillo (15)
            # En un Lunge perfecto visto de perfil, buscamos verticalidad del torso
            torso_lunge = get_angle(kp[5], kp[11])
            
            st.metric("Verticalidad Torso", f"{torso_lunge:.1f}°")
            
            if torso_lunge < 7:
                st.success("SCORE FMS: 3 - Alineación perfecta")
            elif torso_lunge < 20:
                st.warning("SCORE FMS: 2 - Inclinación anterior detectada")
            else:
                st.error("SCORE FMS: 1 - Red Flag: Incapacidad de mantener el torso vertical")

    else:
        st.error("No se detectaron suficientes puntos anatómicos. Asegúrate de que el paciente esté de cuerpo completo en la imagen.")
