import argparse, json
from pathlib import Path
import joblib, torch
from src.pipeline import MLP, norm_text
import pandas as pd

p=argparse.ArgumentParser(); p.add_argument('--query',required=True); p.add_argument('--title',required=True); p.add_argument('--model-dir',default='outputs/models'); a=p.parse_args(); d=Path(a.model_dir)
meta=json.loads((d/'model_metadata.json').read_text()); fb=joblib.load(d/'features.joblib'); m=MLP(meta['input_dim']); m.load_state_dict(torch.load(d/'mlp.pt',map_location='cpu',weights_only=True)); m.eval()
df=pd.DataFrame([{'query':norm_text(a.query),'product_title':norm_text(a.title),'query_words':len(norm_text(a.query).split()),'title_words':len(norm_text(a.title).split())}]); x=fb.transform(df)
with torch.no_grad(): prob=torch.softmax(m(torch.tensor(x)),1).numpy()[0]
print(json.dumps({'predicted_label':meta['labels'][int(prob.argmax())],'probabilities':dict(zip(meta['labels'],map(float,prob)))},indent=2))
