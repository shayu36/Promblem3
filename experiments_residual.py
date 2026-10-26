"""残差 GRU（持久化 + 学习修正）与多种子融合实验。"""
import numpy as np
import torch
import torch.nn as nn

from src.data_utils import load_and_clean, build_windows, split_chronological, scale_data
from src.train import train_model
from src.evaluate import regression_metrics
from config import TARGET_COL, WINDOW_SIZE


class ResidualGRU(nn.Module):
    """输出 = 最后一个时间步的 OT（持久化）+ 网络学到的修正量。"""

    def __init__(self, input_dim, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        last = x[:, -1, -1]                    # 最后时刻的 OT（标准化）
        out, _ = self.gru(x)
        correction = self.head(out[:, -1, :]).squeeze(-1)
        return last + correction


def prepare(df, feat_cols):
    X, y = build_windows(df, feat_cols, WINDOW_SIZE)
    tr, va, te, *_ = split_chronological(len(y))
    ti = feat_cols.index(TARGET_COL)
    Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
        X[tr], X[va], X[te], y[tr], y[va], y[te], ti)
    return Xtr, Xva, Xte, ytr, yva, yte, mu, sd


def run(model, seed):
    m, _ = train_model(model, Xtr, ytr, Xva, yva, verbose=False, seed=seed)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).numpy() * sd + mu


df, fc = load_and_clean()
Xtr, Xva, Xte, ytr, yva, yte, mu, sd = prepare(df, fc)
y_true = yte * sd + mu

print("=" * 60)
# 1) 普通 GRU 基线
from src.models import RecurrentRegressor
p = run(RecurrentRegressor(Xtr.shape[-1], cell="gru"), 42)
m = regression_metrics(y_true, p)
print(f"GRU 基线         RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

# 2) 残差 GRU
p = run(ResidualGRU(Xtr.shape[-1]), 42)
m = regression_metrics(y_true, p)
print(f"残差 GRU         RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

# 3) 残差 GRU 多种子融合
print("-" * 60)
ps = []
for s in [0, 1, 2, 3, 4]:
    p = run(ResidualGRU(Xtr.shape[-1]), s)
    ps.append(p)
    m = regression_metrics(y_true, p)
    print(f"  残差 GRU seed {s}  RMSE={m['RMSE']:.4f} R2={m['R2']:.4f}")
me = regression_metrics(y_true, np.mean(ps, axis=0))
print(f"  5-seed 融合      RMSE={me['RMSE']:.4f} MAE={me['MAE']:.4f} R2={me['R2']:.4f}")
