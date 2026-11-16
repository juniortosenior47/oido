import numpy as np

class AnalizadorAudio:
    def __init__(self, sample_rate, n_fft, freq_min, freq_max, frecuencias_all):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.frecuencias_all = frecuencias_all
        
        # Precalcular índices del rango vocal de interés
        self.indices_rango = np.where((frecuencias_all >= freq_min) & (frecuencias_all <= freq_max))[0]
        self.frecuencias_x = frecuencias_all[self.indices_rango]
        self.n_barras = len(self.frecuencias_x)

    def analizar_frame(self, canal_audio, umbral_silencio):
        """Devuelve métricas por banda para un bloque de audio mono normalizado."""
        if len(canal_audio) == 0:
            ceros = np.zeros(self.n_barras)
            return -np.inf, ceros, ceros.copy(), np.full(self.n_barras, -100.0), ceros.copy(), ceros.copy(), 0.0

        # 1. Volumen RMS Global
        rms_global = np.sqrt(np.mean(canal_audio**2))
        db_global = -60.0 if rms_global < 1e-5 else 20 * np.log10(rms_global)
        
        # 2. Tasa de Cruces por Cero (ZCR) Global
        cruces = canal_audio[:-1] * canal_audio[1:] < 0
        zcr_global = np.mean(cruces) if len(cruces) > 0 else 0.0

        # Vectores de salida vacíos en caso de silencio
        out_pitch = np.zeros(self.n_barras)
        out_brillo = np.zeros(self.n_barras)
        # En dBFS, el silencio debe permanecer en el límite inferior del gráfico.
        out_volumen = np.full(self.n_barras, -100.0)
        out_zcr = np.zeros(self.n_barras)
        out_energia = np.zeros(self.n_barras)

        if db_global < umbral_silencio:
            return db_global, out_pitch, out_brillo, out_volumen, out_zcr, out_energia, 0.0

        # 3. FFT con ventana Hann. El zero-padding suaviza la visualización,
        # pero la normalización mantiene las magnitudes independientes del bloque.
        ventana = np.hanning(len(canal_audio))
        ganancia_coherente = np.sum(ventana)
        fft_raw = np.abs(np.fft.rfft(canal_audio * ventana, n=self.n_fft)) * 2.0 / max(ganancia_coherente, 1e-12)
        out_energia = fft_raw[self.indices_rango]

        # Frecuencia fundamental por autocorrelación. A diferencia del pico FFT,
        # conserva la fundamental incluso cuando un armónico es más energético.
        if len(out_energia) > 0 and np.max(out_energia) > 0:
            indice_max = np.argmax(out_energia)
            max_energia = out_energia[indice_max]
            senal_centrada = canal_audio - np.mean(canal_audio)
            autocorrelacion = np.correlate(senal_centrada, senal_centrada, mode="full")[len(senal_centrada) - 1:]
            retardo_min = max(1, round(self.sample_rate / min(1100, self.freq_max)))
            retardo_max = min(len(autocorrelacion) - 1, round(self.sample_rate / self.freq_min))
            segmento = autocorrelacion[retardo_min:retardo_max + 1]
            picos_locales = np.flatnonzero((segmento[1:-1] > segmento[:-2]) & (segmento[1:-1] >= segmento[2:])) + 1
            if len(picos_locales) > 0:
                retardo = retardo_min + picos_locales[np.argmax(segmento[picos_locales])]
                frecuencia_fundamental = self.sample_rate / retardo
            else:
                frecuencia_fundamental = self.frecuencias_x[indice_max]
        else:
            frecuencia_fundamental = 0
            max_energia = 1e-5

        # Ponderación vectorizada por banda.
        energia_relativa = out_energia / (max_energia + 1e-12)
        cerca_del_pico = np.abs(self.frecuencias_x - frecuencia_fundamental) < 60
        out_pitch = np.where(cerca_del_pico, frecuencia_fundamental * energia_relativa, 0.0)
        out_brillo = np.where(out_energia > 1e-4, self.frecuencias_x * energia_relativa, 0.0)
        out_volumen = np.clip(20 * np.log10(out_energia + 1e-12), -100.0, 0.0)
        out_zcr = zcr_global * energia_relativa
        centroide_espectral = float(np.sum(self.frecuencias_x * out_energia) / (np.sum(out_energia) + 1e-12))

        return db_global, out_pitch, out_brillo, out_volumen, out_zcr, out_energia, centroide_espectral
