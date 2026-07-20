import sys
import time
from datetime import datetime
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# Importaciones modulares locales
from dsp import AnalizadorAudio
from storage import GestorArchivos
from interface import InterfazCyberpunk

# =========================================================================
# CONFIGURACIÓN GENERAL RECALIBRABLE
# =========================================================================
FREQ_MIN = 50      
FREQ_MAX = 3000    
SAMPLE_RATE = 44100
DURATION = 0.015  # Ventana de 15ms para máxima sensibilidad temporal

FACTOR_PADDING = 8  
WINDOW_SIZE = int(SAMPLE_RATE * DURATION)
N_FFT = WINDOW_SIZE * FACTOR_PADDING

UMBRAL_SILENCIO_DB = -50.0       
UMBRAL_PAUSA_FUERTE_DB = -8.0   

programa_pausado = False
esta_grabando = False
bloque_datos_grabados = []
tiempo_inicio = 0.0

# Precalcular base frecuencial espectral
todas_las_frecuencias = np.fft.rfftfreq(N_FFT, d=1 / SAMPLE_RATE)
indices_rango = np.where((todas_las_frecuencias >= FREQ_MIN) & (todas_las_frecuencias <= FREQ_MAX))
frecuencias_x = todas_las_frecuencias[indices_rango]
marcas_hz = np.arange(100, FREQ_MAX + 1, 300)
ancho_barra = (FREQ_MAX - FREQ_MIN) / len(frecuencias_x) * 0.85

# Instanciar Procesador DSP e Interfaz Gráfica
analizador = AnalizadorAudio(SAMPLE_RATE, N_FFT, FREQ_MIN, FREQ_MAX, todas_las_frecuencias)
ui = InterfazCyberpunk(FREQ_MIN, FREQ_MAX, frecuencias_x, marcas_hz, ancho_barra)

# Declaración de Botones en Memoria Global
ax_grabar = plt.axes([0.20, 0.03, 0.15, 0.04])
ax_parar = plt.axes([0.40, 0.03, 0.15, 0.04])
ax_guardar = plt.axes([0.60, 0.03, 0.18, 0.04])

btn_grabar = Button(ax_grabar, 'Grabar 🔴', color='#1f4e3d', hovercolor='#2ea45a')
btn_parar = Button(ax_parar, 'Detener ⏹️', color='#4e2a2a', hovercolor='#a43a3a')
btn_guardar = Button(ax_guardar, 'Guardar CSV 💾', color='#2b4c6f', hovercolor='#3a70a4')

for b in [btn_grabar, btn_parar, btn_guardar]:
    b.label.set_fontsize(10)
    b.label.set_weight('bold')
    b.label.set_color('white')

# =========================================================================
# LÓGICA DE CONTROL DE EVENTOS
# =========================================================================
def presionar_grabar(event):
    global esta_grabando, tiempo_inicio
    esta_grabando = True
    tiempo_inicio = time.time()  # Establecer el punto cero del cronómetro relativo
    ui.set_titulo("🔴 GRABANDO SEÑAL ACÚSTICA MULTIVARIABLE...", color="#FF5555")
    print("[LOG] Grabación iniciada. Cronómetro fijado en 0 milisegundos.")

def presionar_parar(event):
    global esta_grabando
    esta_grabando = False
    ui.set_titulo("Laboratorio Espectral - Grabación en Pausa", color="#FFB86C")
    print(f"[LOG] Grabación en pausa. Muestras en búfer: {len(bloque_datos_grabados)}")

def presionar_guardar(event):
    global bloque_datos_grabados
    exito = GestorArchivos.guardar_grabacion("grabacion_laboratorio.csv", bloque_datos_grabados, frecuencias_x)
    if exito:
        bloque_datos_grabados = []  # Resetear el búfer de memoria tras guardar con éxito
        ui.set_titulo("Laboratorio Espectral - Grabador de Mantisas en Tiempo Real")

def presionar_tecla_espacio(event):
    global programa_pausado
    if event.key == " ":
        programa_pausado = False
        ui.set_titulo("Laboratorio Espectral - Grabador de Mantisas en Tiempo Real")

# Enlazar los botones globales con sus acciones
btn_grabar.on_clicked(presionar_grabar)
btn_parar.on_clicked(presionar_parar)
btn_guardar.on_clicked(presionar_guardar)
ui.fig.canvas.mpl_connect("key_press_event", presionar_tecla_espacio)

# =========================================================================
# FLUJO DE CAPTURA DEL MICRÓFONO
# =========================================================================
def callback_audio(indata, frames, time_info, status):
    global programa_pausado, esta_grabando, bloque_datos_grabados, tiempo_inicio
    
    if programa_pausado:
        return

    canal_audio = indata[:, 0]
    db_g, p_v, b_v, v_v, z_v, e_v = analizador.analizar_frame(canal_audio, UMBRAL_SILENCIO_DB, UMBRAL_PAUSA_FUERTE_DB)

    if db_g > UMBRAL_PAUSA_FUERTE_DB:
        programa_pausado = True
        ui.mostrar_congelado()
        print("[ALERTA] Pico de energía detectado. Pantalla congelada.")
        return

    if esta_grabando and db_g > UMBRAL_SILENCIO_DB:
        # Captura síncrona de los dos ejes de tiempo en el milisegundo exacto de la toma
        tiempo_relativo_seg = time.time() - tiempo_inicio
        marca_absoluta_now = datetime.now()
        
        bloque_datos_grabados.append([
            p_v.copy(), b_v.copy(), v_v.copy(), z_v.copy(), e_v.copy(), 
            tiempo_relativo_seg, marca_absoluta_now
        ])

    ui.actualizar_barras(p_v, b_v, v_v, z_v, e_v)


# Iniciar Stream de Audio
try:
    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=WINDOW_SIZE, callback=callback_audio):
        print("="*60)
        print(" SISTEMA DE TIEMPO DUAL SCONTRONÓMETRO DE MILISEGUNDOS")
        print(" -> Columna 5: Cronómetro relativo puro en enteros (ms).")
        print(" -> Columna 6: Fecha, hora y milisegundos absolutos del día.")
        print("="*60)
        plt.show(block=True)
except KeyboardInterrupt:
    print("\nPrograma finalizado de forma segura.")
    sys.exit()
