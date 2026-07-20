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

    def analizar_frame(self, canal_audio, umbral_silencio, umbral_pausa):
        # 1. Volumen RMS Global
        rms_global = np.sqrt(np.mean(canal_audio**2))
        db_global = -60.0 if rms_global < 1e-5 else 20 * np.log10(rms_global)
        
        # 2. Tasa de Cruces por Cero (ZCR) Global
        cruces = np.diff(np.sign(canal_audio)) != 0
        zcr_global = np.mean(cruces) if len(cruces) > 0 else 0.0

        # Vectores de salida vacíos en caso de silencio
        out_pitch = np.zeros(self.n_barras)
        out_brillo = np.zeros(self.n_barras)
        out_volumen = np.zeros(self.n_barras)
        out_zcr = np.zeros(self.n_barras)
        out_energia = np.zeros(self.n_barras)

        if db_global < umbral_silencio:
            return db_global, out_pitch, out_brillo, out_volumen, out_zcr, out_energia

        # 3. FFT de Alta Resolución (Transitorios puros sin ventanas)
        fft_raw = np.abs(np.fft.rfft(canal_audio, n=self.n_fft))
        out_energia = fft_raw[self.indices_rango]

        # Determinar pico dominante (Pitch)
        if len(out_energia) > 0 and np.max(out_energia) > 0:
            indice_max = np.argmax(out_energia)
            frecuencia_fundamental = self.frecuencias_x[indice_max]
            max_energia = out_energia[indice_max]
        else:
            frecuencia_fundamental = 0
            max_energia = 1e-5

        # Ponderación multi-banda para construir mantisas armónicas
        for i, f in enumerate(self.frecuencias_x):
            peso_energia = out_energia[i]
            
            # Perfil de Pitch
            if abs(f - frecuencia_fundamental) < 60:
                out_pitch[i] = frecuencia_fundamental * (peso_energia / (max_energia + 1e-5))
                
            # Perfil de Brillo
            out_brillo[i] = f * (peso_energia / (max_energia + 1e-5)) if peso_energia > 0.5 else 0
            
            # Perfil de Volumen corregido para base matemática bottom=-50
            db_barra = 20 * np.log10(peso_energia + 1e-4)
            db_barra = max(-50.0, min(0.0, db_barra))  
            out_volumen[i] = db_barra + 50.0            

            # Perfil de ZCR
            out_zcr[i] = zcr_global * (peso_energia / (max_energia + 1e-5))

        return db_global, out_pitch, out_brillo, out_volumen, out_zcr, out_energia