"""Escucha y pinta una palabra usando únicamente el modelo OAF activo."""
import json, os, sys
from pathlib import Path
import tkinter as tk
import numpy as np, psycopg, sounddevice as sd
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'oido-asr-core'))
from oaf import extract, distance
load_dotenv(Path(__file__).with_name('.env')); DB=os.environ['DATABASE_URL']
with psycopg.connect(DB) as c, c.cursor() as q:
 q.execute("SELECT id,model_path,weights FROM oaf_models WHERE is_active ORDER BY created_at DESC LIMIT 1"); model=q.fetchone()
if not model: raise SystemExit('Entrene y active un modelo OAF antes de escuchar.')
model_id,path,weights=model; data=np.load(path); labels=data['labels']; prototypes=data['prototypes']; threshold=float(weights['confidence_threshold'])
root=tk.Tk(); root.title('Escuchar — modelo OAF'); root.geometry('700x250'); text=tk.StringVar(value='Pulse Escuchar')
tk.Label(root,textvariable=text,font=('Segoe UI',28),wraplength=650).pack(expand=True)
def listen():
 audio=sd.rec(16000, samplerate=16000, channels=1, dtype='float32'); sd.wait(); f=extract(audio[:,0]); ds=np.array([distance(f,p) for p in prototypes]); i=int(ds.argmin()); confidence=float(1/(1+ds[i]))
 word=str(labels[i]) if confidence>=threshold else '[no reconocido]'; text.set(f'{word}\nConfianza: {confidence:.0%}')
 with psycopg.connect(DB) as c, c.cursor() as q: q.execute("INSERT INTO oaf_recognition_events(model_id,predicted_label,confidence) VALUES(%s,%s,%s)",(model_id,word,confidence))
tk.Button(root,text='Escuchar 1 segundo',command=listen,font=('Segoe UI',16)).pack(pady=20); root.mainloop()
