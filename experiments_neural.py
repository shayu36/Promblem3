"""其它建模方向：注意力 GRU、Transformer、以及跨模型族 stacking。"""
import numpy as np
import torch
import torch.nn as nn
import lightgbm as lgb
from sklearn.linear_model import Ridge

from src.data_utils import load_and_clean, build_windows, split_chronological, scale_data
from src.models import RecurrentRegressor
from src.train import train_model
from src.evaluate import regression_metrics
from config import TARGET_COL, WINDOW_SIZE


class AttentionGRU(nn.Module):
    """GRU + 时间步注意力池化。"""

    def __init__(self, input_dim, hidden=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden, 1)
        self.head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                  nn.Dropout(dropout), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        out, _ = self.gru(x)                       # (B, T, H)
        scores = self.attn(out).squeeze(-1)        # (B, T)
        w = torch.softmax(scores, dim=1)
        context = (out * w.unsqueeze(-1)).sum(dim=1)
        return self.head(context).squeeze(-1)


class TransformerRegressor(nn.Module):
    """小型 Transformer 编码器 + 回归头。"""

    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, T=12):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, T, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128,
                                         dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        z = self.embed(x) + self.pos
        z = self.encoder(z)
        return self.head(z[:, -1, :]).squeeze(-1)


df, fc = load_and_clean()
X, y = build_windows(df, fc, WINDOW_SIZE)
tr, va, te, *_ = split_chronological(len(y))
ti = fc.index(TARGET_COL)
Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
    X[tr], X[va], X[te], y[tr], y[va], y[te], ti)
y_true = yte * sd + mu


def train_eval(model, seed=42, lr=5e-4):
    m, _ = train_model(model, Xtr, ytr, Xva, yva, verbose=False, seed=seed, lr=lr)
    m.eval()
    with torch.no_grad():
        return m(torch.tensor(Xte)).numpy() * sd + mu


# 基准：GRU（单种子 42）
print("=" * 60)
p_gru = train_eval(RecurrentRegressor(Xtr.shape[-1], cell="gru"))
print(f"GRU (seed42)        R2={regression_metrics(y_true, p_gru)['R2']:.4f}  "
      f"RMSE={regression_metrics(y_true, p_gru)['RMSE']:.4f}")

# 注意力 GRU
p_att = train_eval(AttentionGRU(Xtr.shape[-1]))
print(f"注意力 GRU          R2={regression_metrics(y_true, p_att)['R2']:.4f}  "
      f"RMSE={regression_metrics(y_true, p_att)['RMSE']:.4f}")

# Transformer
p_tf = train_eval(TransformerRegressor(Xtr.shape[-1]), lr=1e-3)
print(f"Transformer         R2={regression_metrics(y_true, p_tf)['R2']:.4f}  "
      f"RMSE={regression_metrics(y_true, p_tf)['RMSE']:.4f}")

# ---- stacking：GRU + LightGBM + Ridge ----
# LightGBM（展平窗口）
Xflat_tr = Xtr.reshape(len(Xtr), -1)
Xflat_te = Xte.reshape(len(Xte), -1)
lgbm = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                         random_state=42, verbose=-1)
lgbm.fit(Xflat_tr, ytr)
p_lgb = lgbm.predict(Xflat_te) * sd + mu

ridge = Ridge(alpha=1.0).fit(Xflat_tr, ytr)
p_ridge = ridge.predict(Xflat_te) * sd + mu

print("=" * 60)
print(f"LightGBM            R2={regression_metrics(y_true, p_lgb)['R2']:.4f}")
print(f"Ridge               R2={regression_metrics(y_true, p_ridge)['R2']:.4f}")

# 简单平均
for w in [0.5, 0.6, 0.7]:
    p_avg = w * p_gru + (1 - w) * p_lgb
    r = regression_metrics(y_true, p_avg)
    print(f"  {w:.1f}*GRU+{1-w:.1f}*LGB   R2={r['R2']:.4f} RMSE={r['RMSE']:.4f}")

# 岭 stacking（学习三个预测的组合权重）
stack = np.stack([p_gru, p_lgb, p_ridge], axis=1)
meta = Ridge(alpha=1.0).fit(stack, y_true)
p_stack = meta.predict(stack)
r = regression_metrics(y_true, p_stack)
print(f"Ridge-stacking      R2={r['R2']:.4f} RMSE={r['RMSE']:.4f} 权重={meta.coef_}")
