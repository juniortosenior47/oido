"""Crea prototipos OAF con muestras aprobadas. Ejecútelo solo cuando decida entrenar."""
import json, os, sys
from pathlib import Path
import numpy as np, psycopg
from dotenv import load_dotenv
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'oido-asr-core'))
from oaf import resample, FEATURE_VERSION
load_dotenv(Path(__file__).with_name('.env')); DB=os.environ['DATABASE_URL']; OUT=Path(__file__).with_name('models'); OUT.mkdir(exist_ok=True)
with psycopg.connect(DB) as c, c.cursor() as q:
 q.execute("SELECT l.label,s.feature_path FROM oaf_samples s JOIN word_labels l ON l.id=s.label_id WHERE s.status='approved' ORDER BY l.label")
 rows=q.fetchall()
 if not rows: raise SystemExit('No hay muestras aprobadas.')
 groups={}
 for label,path in rows: groups.setdefault(label,[]).append(resample(np.load(path)['features']))
 labels=sorted(groups); prototypes=np.stack([np.mean(groups[x],axis=0) for x in labels])
 path=OUT/'oaf-prototypes-v1.npz'; np.savez_compressed(path,labels=np.array(labels),prototypes=prototypes)
 q.execute("INSERT INTO oaf_training_runs(feature_version,sample_count,parameters) VALUES(%s,%s,%s::jsonb) RETURNING id",(FEATURE_VERSION,len(rows),json.dumps({'frames':40,'method':'mean_prototype'}))); run=q.fetchone()[0]
 q.execute("UPDATE oaf_models SET is_active=false WHERE is_active")
 q.execute("INSERT INTO oaf_models(training_run_id,model_path,weights) VALUES(%s,%s,%s::jsonb)",(run,str(path),json.dumps({'distance':1.0,'confidence_threshold':0.60})))
print(f'Modelo entrenado con {len(rows)} muestras: {path}')
