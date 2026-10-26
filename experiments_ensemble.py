"""多种子 GRU 集成：平均多个随机种子的预测以降低方差。"""
import numpy as np
import torch

from src.data_utils import load_and_clean, build_windows, split_chronological, scale_data
from src.models import RecurrentRegressor
from src.train import train_model
from src.evaluate import regression_metrics
from config import TARGET_COL, WINDOW_SIZE


df, fc = load_and_clean()
X, y = build_windows(df, fc, WINDOW_SIZE)
tr, va, te, *_ = split_chronological(len(y))
ti = fc.index(TARGET_COL)
Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
    X[tr], X[va], X[te], y[tr], y[va], y[te], ti)
y_true = yte * sd + mu

seeds = [0, 1, 2, 3, 4]
ps = []
for s in seeds:
    m = RecurrentRegressor(Xtr.shape[-1], cell="gru")
    m, _ = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=s)
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xte)).numpy() * sd + mu
    ps.append(p)
    r = regression_metrics(y_true, p)
    print(f"GRU seed {s}: RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f} R2={r['R2']:.4f}")

mean_p = np.mean(ps, axis=0)
median_p = np.median(ps, axis=0)
r = regression_metrics(y_true, mean_p)
print(f"5-seed 平均融合 : RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f} R2={r['R2']:.4f}")
r = regression_metrics(y_true, median_p)
print(f"5-seed 中位数融合: RMSE={r['RMSE']:.4f} MAE={r['MAE']:.4f} R2={r['R2']:.4f}")
