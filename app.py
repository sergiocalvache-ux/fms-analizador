import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

# 1. Configuración Médica de la App
st.set_page_config(page_title="FMS Analizador Pro", layout="centered")

# --- INICIALIZACIÓN DE LA BASE DE DATOS TEMPORAL (INFORME) ---
if 'informe_clinico' not in st.session_state:
    st.session_state.informe_clinico = {}

# Sidebar para gestión de datos
st.sidebar.title("🩺 Gestión de Sesión")
nombre_paciente = st.sidebar.text_input("Nombre del Paciente", "Paciente Genérico")
test_seleccionado = st.sidebar.selectbox(
    "Seleccionar Test FMS",
    ["Deep Squat", "Hurdle Step", "Inline Lunge"]
)

if st.sidebar.button("🗑️ Limpiar Sesión/Nuevo Paciente"):
    st.session_state.informe_clinico = {}
    st.rerun()

st.title(f"Evaluación FMS: {test_seleccionado}")

# 2. Cargar el motor de IA
@st.cache_resource
def load_model():
    return YOLO('yolov8n-pose.pt')

model = load_model()

# 3. Interfaz de Captura
img_file_buffer = st.camera_input("Capturar Movimiento")

def get_angle(p1, p2):
    return np.degrees(np.arctan2(abs(p1[0]-p2[0]), abs(p1[1]-p2[1])))

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    kp = results[0].keypoints.xy[0].cpu().numpy()

    if len(kp) > 15:
        st.image(annotated_frame, caption=f"Resultado: {test_seleccionado}")
        
        # Lógica de cálculo (resumida para el informe)
        diagnostico = ""
        score = 0
        
        if test_seleccionado == "Deep Squat":
            ang_torso = get_angle(kp[5], kp[11])
            ang_tibia = get_angle(kp[13], kp[15])
            dif = abs(ang_torso - ang_tibia)
            score = 3 if dif < 10 else 2 if dif < 25 else 1
            diagnostico = f"Torso: {ang_torso:.1f}°, Tibia: {ang_tibia:.1f}°. Dif: {dif:.1f}°."
            pautas = "Trabajar movilidad de tobillo (dorsiflexión) y control de core." if score < 3 else "Patrón óptimo."

        elif test_seleccionado == "Hurdle Step":
            verticalidad = get_angle(kp[5], kp[11])
            score = 3 if verticalidad < 5 else 2 if verticalidad < 15 else 1
            diagnostico = f"Inclinación de tronco: {verticalidad:.1f}°."
            pautas = "Enfoque en estabilidad lumbo-pélvica y fuerza de glúteo medio." if score < 3 else "Estabilidad excelente."

        elif test_seleccionado == "Inline Lunge":
            torso_lunge = get_angle(kp[5], kp[11])
            score = 3 if torso_lunge < 7 else 2 if torso_lunge < 20 else 1
            diagnostico = f"Verticalidad del torso: {torso_lunge:.1f}°."
            pautas = "Mejorar estabilidad monopodal y movilidad de cadera (psoas)." if score < 3 else "Alineación correcta."

        # ALMACENAR EN EL INFORME
        if st.button(f"✅ Guardar {test_seleccionado} en Informe"):
            st.session_state.informe_clinico[test_seleccionado] = {
                "score": score,
                "detalles": diagnostico,
                "pautas": pautas,
                "imagen": annotated_frame
            }
            st.success(f"{test_seleccionado} guardado con éxito.")

# --- SECCIÓN DEL INFORME FINAL ---
st.divider()
st.header("📋 Informe Consolidado")

if st.session_state.informe_clinico:
    for test, datos in st.session_state.informe_clinico.items():
        with st.expander(f"Resultado {test} - Score: {datos['score']}"):
            st.write(f"**Análisis:** {datos['detalles']}")
            st.write(f"**Pautas sugeridas:** {datos['pautas']}")
            st.image(datos['imagen'], width=300)
    
    # Botón para preparar texto para Word
    if st.button("📝 Generar Texto para Informe"):
        texto_informe = f"INFORME FMS - PACIENTE: {nombre_paciente}\n" + "="*30 + "\n"
        for test, datos in st.session_state.informe_clinico.items():
            texto_informe += f"\nTEST: {test}\nScore: {datos['score']}\nDatos: {datos['detalles']}\nPautas: {datos['pautas']}\n"
        
        st.text_area("Copia esto en tu Word:", texto_informe, height=200)
else:
    st.info("Aún no hay pruebas guardadas en esta sesión.")
