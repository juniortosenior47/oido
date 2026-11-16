import csv

class GestorArchivos:
    @staticmethod
    def guardar_grabacion(nombre_archivo, bloque_datos, frecuencias_x):
        if not bloque_datos:
            print("[ALERTA] El búfer de grabación está vacío.")
            return False
        
        print(f"[CSV] Exportando {len(bloque_datos)} muestras en alta precisión temporal...")
        try:
            with open(nombre_archivo, "w", newline="", encoding="utf-8") as f:
                escritor = csv.writer(f)
                # Encabezado corregido con las dos columnas temporales solicitadas
                escritor.writerow(["Muestra_ID", "Tipo_Grafica", "Frecuencia_Hz", "Valor_Medido", "Tiempo_Grabacion_Mseg", "Fecha_Hora_Mseg"])
                
                for idx, muestra in enumerate(bloque_datos):
                    p_v, b_v, v_v, z_v, e_v, tiempo_relativo_seg, marca_absoluta, *metricas_globales = muestra
                    rms_db, centroide_hz = metricas_globales if metricas_globales else (None, None)
                    
                    # Convertir segundos relativos a milisegundos enteros (ej: 0.015 -> 15)
                    tiempo_mseg = int(round(tiempo_relativo_seg * 1000))
                    
                    # Formatear la fecha y hora absoluta incluyendo milisegundos (.fff)
                    tiempo_absoluto_formato = marca_absoluta.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    for i, hz in enumerate(frecuencias_x):
                        escritor.writerow([idx, "1_Pico_Dominante", f"{hz:.1f}", f"{p_v[i]:.2f}", tiempo_mseg, tiempo_absoluto_formato])
                        escritor.writerow([idx, "2_Brillo", f"{hz:.1f}", f"{b_v[i]:.2f}", tiempo_mseg, tiempo_absoluto_formato])
                        escritor.writerow([idx, "3_Volumen_dBFS", f"{v_v[i]:.2f}", tiempo_mseg, tiempo_absoluto_formato])
                        escritor.writerow([idx, "4_ZCR", f"{hz:.1f}", f"{z_v[i]:.4f}", tiempo_mseg, tiempo_absoluto_formato])
                        escritor.writerow([idx, "5_Energia", f"{hz:.1f}", f"{e_v[i]:.6f}", tiempo_mseg, tiempo_absoluto_formato])
                    if rms_db is not None:
                        escritor.writerow([idx, "6_RMS_Global_dBFS", "", f"{rms_db:.2f}", tiempo_mseg, tiempo_absoluto_formato])
                        escritor.writerow([idx, "7_Centroide_Espectral_Hz", "", f"{centroide_hz:.2f}", tiempo_mseg, tiempo_absoluto_formato])
                        
            print(f"[CSV] ¡Éxito! Archivo '{nombre_archivo}' guardado de forma segura.")
            return True
        except Exception as e:
            print(f"[ERROR] No se pudo escribir el archivo: {e}")
            return False
