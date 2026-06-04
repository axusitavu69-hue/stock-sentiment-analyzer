"""纯净训练子进程 — 只加载numpy+lightgbm，不碰akshare"""
import sys, os, pickle, json, numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Read training data passed via pickle
data = pickle.load(open(sys.argv[1], 'rb'))
X, y = data['X'], data['y']
tag = data.get('tag', 'train')
samples = data.get('samples', len(X))
stocks = data.get('stocks', 0)

split = int(len(X) * 0.8)
X_tr, X_val = X[:split], X[split:]
y_tr, y_val = y[:split], y[split:]

n = len(X)
if n < 5000: n_est, n_leaves, lr = 100, 15, 0.05
elif n < 50000: n_est, n_leaves, lr = 300, 31, 0.02
elif n < 200000: n_est, n_leaves, lr = 500, 47, 0.01
else: n_est, n_leaves, lr = 800, 63, 0.008

scale = (1 - y_tr.mean()) / (y_tr.mean() + 1e-9)
model = LGBMClassifier(n_estimators=n_est, learning_rate=lr, num_leaves=n_leaves,
    max_depth=7, min_child_samples=max(5,min(30,n//500)),
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
    scale_pos_weight=scale, random_state=42, n_jobs=-1, verbose=-1)

model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(stopping_rounds=50, verbose=False), log_evaluation(period=-1)])

acc = float(model.score(X_val, y_val))
imp = model.feature_importances_
preds = model.predict(X_val)
rpt = classification_report(y_val, preds, output_dict=True, zero_division=0)
f1_up = rpt.get('1', {}).get('f1-score', 0)
f1_down = rpt.get('0', {}).get('f1-score', 0)

print(f"  [{tag}] 准确率: {acc:.1%} | F1up: {f1_up:.3f} | F1dn: {f1_down:.3f} | {len(X_tr)}/{len(X_val)}")

# Save results
result = {'model': model, 'acc': acc, 'imp': imp, 'f1_up': f1_up, 'f1_down': f1_down}
pickle.dump(result, open(sys.argv[2], 'wb'))
