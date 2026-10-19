import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
base=json.loads((OUT/'metrics.json').read_text()); adv=json.loads((OUT/'advanced_metrics.json').read_text())
split=json.loads((OUT/'split_indices.json').read_text()); raw=pd.read_csv(ROOT/'data'/'fake_job_postings.csv')
y=raw.fraudulent.to_numpy()[np.asarray(split['test'])]; score=np.load(OUT/'advanced_test_scores.npy')
plt.style.use('seaborn-v0_8-whitegrid')
p,r,_=precision_recall_curve(y,score); fig,ax=plt.subplots(figsize=(6.4,4.2)); ax.plot(r,p,color='#183B5B',lw=2)
ax.axhline(raw.fraudulent.mean(),color='#999999',ls='--',label=f"Dataset prevalence {raw.fraudulent.mean():.3f}")
ax.set(xlabel='Recall',ylabel='Precision',title='Advanced ensemble precision recall curve'); ax.legend(loc='lower left'); fig.tight_layout()
fig.savefig(OUT/'figures'/'precision_recall.png',dpi=180); plt.close(fig)
names=['Structured','Word text','Linear fused','Advanced']; vals=[base['models']['structured']['test_selected']['pr_auc'],base['models']['word_text']['test_selected']['pr_auc'],base['models']['fused']['test_selected']['pr_auc'],adv['advanced_ensemble_test']['pr_auc']]
fig,ax=plt.subplots(figsize=(7.2,4.2)); ax.bar(names,vals,color=['#A8B2BC','#6F9ABE','#416B8E','#183B5B']); ax.set_ylim(0,1); ax.set_ylabel('Test PR AUC'); ax.set_title('Model comparison on grouped holdout')
for i,v in enumerate(vals): ax.text(i,v+.022,f'{v:.3f}',ha='center')
fig.tight_layout(); fig.savefig(OUT/'figures'/'model_comparison.png',dpi=180); plt.close(fig)
