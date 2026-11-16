"""Captura un patrón OAF etiquetado; no entrena ni cambia pesos."""
import argparse, os, sys
from pathlib import Path
import numpy as np, psycopg, sounddevice as sd
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'oido-asr-core'))
from oaf import extract, FEATURE_VERSION
load_dotenv(Path(__file__).with_name('.env')); DB=os.environ['DATABASE_URL']; OUT=Path(__file__).with_name('samples')
p=argparse.ArgumentParser(); p.add_argument('--label',required=True); p.add_argument('--seconds',type=float,default=1.5); a=p.parse_args()
audio=sd.rec(int(a.seconds*16000), samplerate=16000, channels=1, dtype='float32'); sd.wait(); features=extract(audio[:,0])
OUT.mkdir(exist_ok=True); path=OUT/f'{a.label}_{len(list(OUT.glob("*.npz"))):05d}.npz'; np.savez_compressed(path, features=features)
with psycopg.connect(DB) as c, c.cursor() as q:
 q.execute("INSERT INTO word_labels(label) VALUES(%s) ON CONFLICT(label) DO UPDATE SET label=EXCLUDED.label RETURNING id",(a.label.lower(),)); label_id=q.fetchone()[0]
 q.execute("INSERT INTO oaf_samples(label_id,feature_version,feature_path,frame_count,feature_count) VALUES(%s,%s,%s,%s,%s)",(label_id,FEATURE_VERSION,str(path),len(features),features.shape[1]))
print(f'Patrón guardado: {path}. Estado: pending; apruébelo antes de entrenar.')
