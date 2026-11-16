"""Crea una nueva configuración; no reentrena ni modifica el historial."""
import argparse, json, os
from pathlib import Path
import psycopg
from dotenv import load_dotenv
load_dotenv(Path(__file__).with_name('.env')); DB=os.environ['DATABASE_URL']
p=argparse.ArgumentParser(); p.add_argument('--threshold',type=float,required=True); a=p.parse_args()
if not 0<=a.threshold<=1: raise SystemExit('El umbral debe estar entre 0 y 1.')
with psycopg.connect(DB) as c, c.cursor() as q:
 q.execute("SELECT training_run_id,model_path,weights FROM oaf_models WHERE is_active ORDER BY created_at DESC LIMIT 1"); row=q.fetchone()
 if not row: raise SystemExit('No existe un modelo activo.')
 run,path,weights=row; weights=dict(weights); weights['confidence_threshold']=a.threshold
 q.execute("UPDATE oaf_models SET is_active=false WHERE is_active")
 q.execute("INSERT INTO oaf_models(training_run_id,model_path,weights) VALUES(%s,%s,%s::jsonb)",(run,path,json.dumps(weights)))
print(f'Nueva versión de pesos creada: umbral={a.threshold:.0%}')
