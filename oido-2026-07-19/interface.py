import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class InterfazCyberpunk:
    def __init__(self, freq_min, freq_max, frecuencias_x, marcas_hz, ancho_barra):
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.frecuencias_x = frecuencias_x
        self.n_barras = len(frecuencias_x)
        
        # Configurar estilos de entorno oscuro nativo
        plt.style.use('dark_background')
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4, self.ax5) = plt.subplots(5, 1, figsize=(14, 14))
        self.fig.patch.set_facecolor('#0B0C10')
        self.fig.suptitle("Laboratorio Espectral - Grabador de Mantisas en Tiempo Real", fontsize=16, fontweight="bold", color="#66FCF1")

        # Inicializar y dar estilo neón a cada panel
        self._estiliza_panel(self.ax1, "1. Espectro de Pitch (Cresta de Nota Base)", "Pitch (Hz)", (0, 1100), np.arange(0, 1101, 200))
        self.barras_tono = self.ax1.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#BD93F9", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax2, "2. Espectro de Brillo (Valles y Crestas de Formantes)", "Brillo (Hz)", (0, freq_max + 100), np.arange(0, freq_max + 1, 500))
        self.barras_espectro = self.ax2.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#FFB86C", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax3, "3. Envolvente de Energía Logarítmica", "Volumen (dB)", (-50, 0), np.arange(-50, 1, 10))
        self.linea_umbral = self.ax3.axhline(-8.0, color="#FF5555", linestyle="--", alpha=0.8, linewidth=1.5)
        self.barras_volumen = self.ax3.bar(frecuencias_x, np.zeros(self.n_barras), bottom=-50.0, width=ancho_barra, color="#FF5555", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax4, "4. Tasa de Cruces por Cero (ZCR)", "ZCR", (0, 0.5), [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        self.barras_zcr = self.ax4.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#8BE9FD", alpha=0.9, edgecolor="none")

        self._estiliza_panel(self.ax5, "5. Espectro de Magnitud Completo (Ecualizador)", "Energía", (0, 50), np.arange(0, 51, 10))
        self.ax5.set_xlim(freq_min, freq_max)
        self.ax5.set_xlabel("Eje de Frecuencias Compartido (Hz)", fontsize=10, fontweight="bold", color="#C5C6C7")
        self.ax5.set_xticks(marcas_hz)
        self.ax5.tick_params(axis='x', rotation=45)
        self.barras_espectro_inst = self.ax5.bar(frecuencias_x, np.zeros(self.n_barras), width=ancho_barra, color="#50FA7B", alpha=0.9, edgecolor="none")

        # Limpieza de textos redundantes intermedios
        plt.setp(self.ax1.get_xticklabels(), visible=False)
        plt.setp(self.ax2.get_xticklabels(), visible=False)
        plt.setp(self.ax3.get_xticklabels(), visible=False)
        plt.setp(self.ax4.get_xticklabels(), visible=False)

        plt.subplots_adjust(bottom=0.15, top=0.92, hspace=0.35)

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
            self.barras_volumen[i].set_height(v_v[i])
            self.barras_zcr[i].set_height(z_v[i])
            self.barras_espectro_inst[i].set_height(e_v[i])

        max_e = np.max(e_v)
        if max_e > 5:
            self.ax5.set_ylim(0, max_e * 1.2)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def mostrar_congelado(self):
        self.fig.suptitle("⚠️ PANTALLA CONGELADA (Sonido Fuerte). Presiona ESPACIO para continuar ⚠️", color="#FF5555", fontsize=14, fontweight="bold")
        self.fig.canvas.draw_idle()

    def set_titulo(self, texto, color="#66FCF1"):
        self.fig.suptitle(texto, color=color, fontsize=16, fontweight="bold")
