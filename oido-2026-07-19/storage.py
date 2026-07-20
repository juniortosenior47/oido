import os
from datetime import datetime

class GestorArchivos:
    @staticmethod
    def guardar_grabacion(nombre_archivo, bloque_datos, frecuencias_x):
        if not bloque_datos:
            print("[ALERTA] El búfer de grabación está vacío.")
            return False
        
        print(f"[CSV] Exportando {len(bloque_datos)} muestras en alta precisión temporal...")
        try:
            with open(nombre_archivo, "w") as f:
                # Encabezado corregido con las dos columnas temporales solicitadas
                f.write("Muestra_ID,Tipo_Grafica,Frecuencia_Hz,Valor_Medido,Tiempo_Grabacion_Mseg,Fecha_Hora_Mseg\n")
                
                for idx, muestra in enumerate(bloque_datos):
                    p_v, b_v, v_v, z_v, e_v, tiempo_relativo_seg, marca_absoluta = muestra
                    
                    # Convertir segundos relativos a milisegundos enteros (ej: 0.015 -> 15)
                    tiempo_mseg = int(round(tiempo_relativo_seg * 1000))
                    
                    # Formatear la fecha y hora absoluta incluyendo milisegundos (.fff)
                    tiempo_absoluto_formato = marca_absoluta.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

                    for i, hz in enumerate(frecuencias_x):
                        f.write(f"{idx},1_Pitch,{hz:.1f},{p_v[i]:.2f},{tiempo_mseg},{tiempo_absoluto_formato}\n")
                        f.write(f"{idx},2_Brillo,{hz:.1f},{b_v[i]:.2f},{tiempo_mseg},{tiempo_absoluto_formato}\n")
                        f.write(f"{idx},3_Volumen,{hz:.1f},{(v_v[i] - 50.0):.2f},{tiempo_mseg},{tiempo_absoluto_formato}\n")
                        f.write(f"{idx},4_ZCR,{hz:.1f},{z_v[i]:.4f},{tiempo_mseg},{tiempo_absoluto_formato}\n")
                        f.write(f"{idx},5_Energia,{hz:.1f},{e_v[i]:.2f},{tiempo_mseg},{tiempo_absoluto_formato}\n")
                        
            print(f"[CSV] ¡Éxito! Archivo '{nombre_archivo}' guardado de forma segura.")
            return True
        except Exception as e:
            print(f"[ERROR] No se pudo escribir el archivo: {e}")
            return False
