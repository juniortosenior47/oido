import matplotlib.pyplot as plt
import numpy as np
from collections import deque

class InterfazCyberpunk:
    def __init__(self, freq_min, freq_max, frecuencias_x, marcas_hz, ancho_barra):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.frecuencias_x = frecuencias_x
        self.n_barras = len(frecuencias_x)
        
        # Configurar estilos de entorno oscuro nativo
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5, self.ax6) = plt.subplots(6, 1, figsize=(14, 16))
        self.fig.patch.set_facecolor('#0B0C10')
        self.fig.suptitle("Laboratorio Espectral - Grabador de Mantisas en Tiempo Real", fontsize=16, fontweight="bold", color="#66FCF1")

        # Inicializar y dar estilo neón a cada panel
        self._estiliza_panel(self.ax1, "1. Pico espectral dominante", "Frecuencia (Hz)", (0, 1100), np.arange(0, 1101, 200))
        self.barras_tono = self.ax1.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#BD93F9", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax2, "2. Espectro de Brillo (Valles y Crestas de Formantes)", "Brillo (Hz)", (0, freq_max + 100), np.arange(0, freq_max + 1, 500))
        self.barras_espectro = self.ax2.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#FFB86C", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax3, "3. Magnitud espectral", "Nivel (dBFS)", (-100, 0), np.arange(-100, 1, 20))
        self.linea_umbral = self.ax3.axhline(-8.0, color="#FF5555", linestyle="--", alpha=0.8, linewidth=1.5)
        self.barras_volumen = self.ax3.bar(frecuencias_x, np.zeros(self.n_barras), bottom=-100.0, width=ancho_barra, color="#FF5555", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax4, "4. Tasa de Cruces por Cero (ZCR)", "ZCR", (0, 0.5), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.barras_zcr = self.ax4.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#8BE9FD", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax5, "5. Espectro de magnitud", "Amplitud", (0, 1), np.arange(0, 1.1, 0.2))
        self.ax5.set_xlim(freq_min, freq_max)
        self.ax5.set_xlabel("Eje de Frecuencias Compartido (Hz)", fontsize=10, fontweight="bold", color="#C5C6C7")
        self.ax5.set_xticks(marcas_hz)
        self.ax5.tick_params(axis='x', rotation=45)
        self.barras_espectro_inst = self.ax5.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#50FA7B", alpha=0.9, edgecolor="none")

        # Historial temporal de métricas globales (ventana móvil de 10 segundos).
        self._estiliza_panel(self.ax6, "6. Evolución temporal: nivel y centroide espectral", "RMS (dBFS)", (-100, 0), np.arange(-100, 1, 20))
        self.ax6.set_xlabel("Tiempo (s)", fontsize=10, fontweight="bold", color="#C5C6C7")
        self.ax6_centroid = self.ax6.twinx()
        self.ax6_centroid.set_ylabel("Centroide (Hz)", fontsize=10, fontweight="bold", color="#FFB86C")
        self.ax6_centroid.set_ylim(freq_min, freq_max)
        self.ax6_centroid.tick_params(colors="#FFB86C", labelsize=9)
        self.historial_rms = deque(maxlen=300)
        self.historial_centroide = deque(maxlen=300)
        self.historial_tiempo = deque(maxlen=300)
        self.ventana_temporal_segundos = 10
        self.ax6.set_xlim(-self.ventana_temporal_segundos, 0)
        self.ax6.set_xticks(np.arange(-self.ventana_temporal_segundos, 1, 2))
        self.linea_rms, = self.ax6.plot([], [], color="#FF5555", linewidth=1.8, label="RMS")
        self.linea_centroide, = self.ax6_centroid.plot([], [], color="#FFB86C", linewidth=1.5, label="Centroide")

        # Limpieza de textos redundantes intermedios
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        plt.setp(self.ax2.get_xticklabels(), visible=False)
        plt.setp(self.ax3.get_xticklabels(), visible=False)
        plt.setp(self.ax4.get_xticklabels(), visible=False)

        plt.subplots_adjust(bottom=0.12, top=0.92, hspace=0.55)

    def _estiliza_panel(self, ax, titulo, ylabel, ylim, yticks):
        ax.set_facecolor('#1F2833')
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color="#C5C6C7")
        ax.set_title(titulo, fontsize=11, loc="left", color="#66FCF1")
        ax.set_yticks(yticks)
        ax.grid(True, linestyle=":", alpha=0.3, color="#66FCF1")
        ax.tick_params(colors='#C5C6C7', labelsize=9)

    def actualizar_barras(self, p_v, b_v, v_v, z_v, e_v):
        for i in range(self.n_barras):
            self.barras_tono[i].set_height(p_v[i])
            self.barras_espectro[i].set_height(b_v[i])
            # La barra parte de -100 dBFS; su altura debe llegar al nivel medido.
            self.barras_volumen[i].set_height(v_v[i] + 100.0)
            self.barras_zcr[i].set_height(z_v[i])
            self.barras_espectro_inst[i].set_height(e_v[i])

        max_e = np.max(e_v)
        self.ax5.set_ylim(0, max(1e-3, max_e * 1.2))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def actualizar_metricas_temporales(self, rms_db, centroide_hz, instante):
        self.historial_rms.append(rms_db)
        self.historial_centroide.append(centroide_hz)
        self.historial_tiempo.append(instante)
        x = np.asarray(self.historial_tiempo) - instante
        self.linea_rms.set_data(x, self.historial_rms)
        self.linea_centroide.set_data(x, self.historial_centroide)
        self.fig.canvas.draw_idle()

    def mostrar_congelado(self):
        self.fig.suptitle("⚠️ PANTALLA CONGELADA (Sonido Fuerte). Presiona ESPACIO para continuar ⚠️", color="#FF5555", fontsize=14, fontweight="bold")
        self.fig.canvas.draw_idle()

    def set_titulo(self, texto, color="#66FCF1"):
        self.fig.suptitle(texto, color=color, fontsize=16, fontweight="bold")
