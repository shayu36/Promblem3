"""任务3 正确 stacking：在验证集上拟合组合权重，测试集上评估（无泄漏）。"""
import numpy as np
import torch
import lightgbm as lgb
from sklearn.linear_model import Ridge

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


def gru_preds(seeds=(0, 1, 2, 3, 4)):
    pv, pt = [], []
    for s in seeds:
        m = RecurrentRegressor(Xtr.shape[-1], cell="gru")
        m, _ = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=s)
        m.eval()
        with torch.no_grad():
            pv.append(m(torch.tensor(Xva)).numpy())
            pt.append(m(torch.tensor(Xte)).numpy())
    return np.mean(pv, axis=0), np.mean(pt, axis=0)


# 展平特征
Xtr_f, Xva_f, Xte_f = Xtr.reshape(len(Xtr), -1), Xva.reshape(len(Xva), -1), Xte.reshape(len(Xte), -1)

# GRU 集成预测（标准化量纲）
gv, gt = gru_preds()
# Ridge（标准化目标）
rv = Ridge(alpha=1.0).fit(Xtr_f, ytr).predict(Xva_f)
rt = Ridge(alpha=1.0).fit(Xtr_f, ytr).predict(Xte_f)
# LightGBM
lv = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                       random_state=42, verbose=-1).fit(Xtr_f, ytr).predict(Xva_f)
lt = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                       random_state=42, verbose=-1).fit(Xtr_f, ytr).predict(Xte_f)

# 单模型（还原量纲）
for name, p in [("GRU集成", gt), ("Ridge", rt), ("LGBM", lt)]:
    r = regression_metrics(yte * sd + mu, p * sd + mu)
    print(f"{name:10s} R2={r['R2']:.4f} RMSE={r['RMSE']:.4f}")

# 在验证集上拟合 stacking 权重
stack_v = np.stack([gv, rv, lv], axis=1)
stack_t = np.stack([gt, rt, lt], axis=1)
meta = Ridge(alpha=1.0).fit(stack_v, yva)
pt = meta.predict(stack_t)
r = regression_metrics(yte * sd + mu, pt * sd + mu)
print(f"Stacking  R2={r['R2']:.4f} RMSE={r['RMSE']:.4f} 权重={np.round(meta.coef_,4)}")

# 简单平均（GRU+Ridge）作为对照
for w in [0.7, 0.8, 0.9]:
    p = w * gt + (1 - w) * rt
    r = regression_metrics(yte * sd + mu, p * sd + mu)
    print(f"{w}*GRU+{1-w}*Ridge  R2={r['R2']:.4f} RMSE={r['RMSE']:.4f}")
