"""模型优化实验：时间特征、双向、更大容量、多种子融合。"""
import numpy as np
import pandas as pd
import torch

from src.data_utils import load_and_clean, build_windows, split_chronological, scale_data
from src.models import RecurrentRegressor
from src.train import train_model
from src.evaluate import regression_metrics
from config import TARGET_COL, WINDOW_SIZE


def add_time_features(df):
    date = pd.to_datetime(df["date"], format="%Y/%m/%d %H:%M")
    hod = date.dt.hour + date.dt.minute / 60.0
    doy = date.dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * hod / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hod / 24)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def train_eval(feat_cols, cell="gru", hidden=128, layers=2, bidir=False, seed=42):
    X, y = build_windows(df, feat_cols, WINDOW_SIZE)
    tr, va, te, *_ = split_chronological(len(y))
    ti = feat_cols.index(TARGET_COL)
    Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
        X[tr], X[va], X[te], y[tr], y[va], y[te], ti)

    m = RecurrentRegressor(Xtr.shape[-1], hidden=hidden, num_layers=layers,
                           dropout=0.2, cell=cell)
    if bidir:
        head_in = hidden * 2
        m.rnn = (torch.nn.LSTM if cell == "lstm" else torch.nn.GRU)(
            input_size=Xtr.shape[-1], hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=0.2 if layers > 1 else 0.0, bidirectional=True)
        m.head = torch.nn.Sequential(
            torch.nn.Linear(head_in, hidden), torch.nn.ReLU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(hidden, 1))

    m, _ = train_model(m, Xtr, ytr, Xva, yva, verbose=False, seed=seed)
    m.eval()
    with torch.no_grad():
        p = m(torch.tensor(Xte)).numpy() * sd + mu
    return regression_metrics(yte * sd + mu, p)


df, fc = load_and_clean()
df = add_time_features(df)
fc_time = fc + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

print("=" * 64)
m = train_eval(fc)
print(f"GRU 21特征(基线)      RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
m = train_eval(fc_time)
print(f"GRU 25特征(+时间)     RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

print("=" * 64)
print("25 特征下结构对比：")
m = train_eval(fc_time, hidden=128, layers=2, bidir=True)
print(f"  GRU 128x2 bidir     RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
m = train_eval(fc_time, hidden=256, layers=2)
print(f"  GRU 256x2           RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
m = train_eval(fc_time, hidden=256, layers=3)
print(f"  GRU 256x3           RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
m = train_eval(fc_time, cell="lstm", hidden=256, layers=2)
print(f"  LSTM 256x2          RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")

print("=" * 64)
print("多种子融合（25 特征 GRU 128x2）：")
ps = []
for s in [0, 1, 2]:
    m = train_eval(fc_time, seed=s)
    ps.append(m)
    print(f"  seed {s}            RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} R2={m['R2']:.4f}")
# 平均各项指标（近似融合效果）
import statistics
avg = {k: float(np.mean([p[k] for p in ps])) for k in ps[0]}
print(f"  3-seed 平均指标      RMSE={avg['RMSE']:.4f} MAE={avg['MAE']:.4f} R2={avg['R2']:.4f}")
