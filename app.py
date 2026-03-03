import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from docx import Document
from docx.shared import Inches
import io

# 1. Configuración de la App
st.set_page_config(page_title="FMS Analizador Pro", layout="centered")

if 'informe_clinico' not in st.session_state:
    st.session_state.informe_clinico = {}

st.sidebar.title("🩺 Gestión de Sesión")
nombre_paciente = st.sidebar.text_input("Nombre del Paciente", "Paciente Genérico")
test_seleccionado = st.sidebar.selectbox(
    "Seleccionar Test FMS",
    ["Deep Squat", "Hurdle Step", "Inline Lunge", "Shoulder Mobility"]
)

# Selector de lado solo para movilidad de hombro
lado = ""
if test_seleccionado == "Shoulder Mobility":
    lado = st.sidebar.radio("Lado a evaluar (mano que sube):", ["Derecha", "Izquierda"])

if st.sidebar.button("🗑️ Nueva Sesión"):
    st.session_state.informe_clinico = {}
    st.rerun()

st.title(f"Evaluación FMS: {test_seleccionado} {lado}")

@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# --- FUNCIONES DE APOYO ---
def get_angle(p1, p2):
    return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))

def get_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def crear_word(datos_informe, nombre):
    doc = Document()
    doc.add_heading(f'Informe de Evaluación FMS', 0)
    doc.add_paragraph(f'Paciente: {nombre}')
    for test_key, datos in datos_informe.items():
        doc.add_heading(f'Test: {test_key}', level=1)
        doc.add_paragraph(f'Score: {datos["score"]}')
        doc.add_paragraph(f'Análisis: {datos["detalles"]}')
        doc.add_paragraph(f'Pautas: {datos["pautas"]}')
        img = Image.fromarray(datos["imagen"])
        img_stream = io.BytesIO()
        img.save(img_stream, format='PNG')
        img_stream.seek(0)
        doc.add_picture(img_stream, width=Inches(4))
    target_stream = io.BytesIO()
    doc.save(target_stream)
    target_stream.seek(0)
    return target_stream

# --- INTERFAZ DE CAPTURA ---
img_file_buffer = st.camera_input("Capturar Movimiento")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    kp = results[0].keypoints.xy[0].cpu().numpy()

    if len(kp) > 15:
        st.image(annotated_frame, caption=f"Análisis {test_seleccionado}")
        
        detalles, pautas, score = "", "", 0
        
        # --- LÓGICA POR TEST ---
        if test_seleccionado == "Deep Squat":
            dif = abs(get_angle(kp[5], kp[11]) - get_angle(kp[13], kp[15]))
            score = 3 if dif < 10 else 2 if dif < 25 else 1
            detalles = f"Diferencia Torso-Tibia: {dif:.1f}°."
            pautas = "Movilidad de tobillo y control motor."

        elif test_seleccionado == "Shoulder Mobility":
            # Distancia entre manos (puntos 9 y 10)
            dist_manos = get_distance(kp[9], kp[10])
            # Estimación tamaño mano (distancia codo-muñeca / 3)
            tam_mano = get_distance(kp[7], kp[9]) / 2.5 
            
            relacion = dist_manos / tam_mano
            score = 3 if relacion < 1.0 else 2 if relacion < 1.5 else 1
            detalles = f"Lado {lado}. Distancia manos: {relacion:.2f} veces el tamaño de la mano."
            pautas = "Movilidad torácica y estiramiento de pectorales/dorsales."

        # (Mantener lógica de Hurdle Step e Inline Lunge aquí...)

        label_guardar = f"{test_seleccionado}_{lado}" if lado else test_seleccionado
        if st.button(f"💾 Guardar {label_guardar}"):
            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.session_state.informe_clinico[label_guardar] = {
                "score": score, "detalles": detalles, "pautas": pautas, "imagen": img_rgb
            }
            st.toast("Captura guardada")

# --- SECCIÓN DE DESCARGA ---
if st.session_state.informe_clinico:
    st.divider()
    doc_word = crear_word(st.session_state.informe_clinico, nombre_paciente)
    st.download_button(label="📥 Descargar Informe Completo (.docx)", data=doc_word, 
                       file_name=f"FMS_{nombre_paciente}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

