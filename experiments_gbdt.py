"""GBDT 优化实验：LightGBM / XGBoost 在滑动窗口 + 时间 + 滞后 + 滚动特征上。"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_utils import load_and_clean, split_chronological
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


def build_features(df, feat_cols, window=WINDOW_SIZE, use_lags=True, use_rolling=True):
    X = df[feat_cols].values.astype(np.float32)
    ot = df[TARGET_COL].values.astype(np.float32)
    tfeat = ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]
    T = df[tfeat].values.astype(np.float32)

    rows, ys = [], []
    for i in range(window, len(df)):
        feats = [X[i - window:i].ravel()]          # 窗口展平 (window * F)
        feats.append(T[i])                          # 目标时刻的时间特征
        if use_lags:
            lags = [1, 6, 12, 24, 72, 144]
            feats.append([ot[i - k] if i >= k else np.nan for k in lags])
        if use_rolling:
            feats.append([ot[i - 6:i].mean(), ot[i - 6:i].std(),
                          ot[i - 12:i].mean(), ot[i - 12:i].std(),
                          ot[max(0, i - 144):i].mean(), ot[max(0, i - 144):i].std()])
        rows.append(np.concatenate([np.atleast_1d(f).ravel() for f in feats]))
        ys.append(ot[i])
    return np.asarray(rows, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def report(y_true, y_pred, name):
    print(f"{name:28s} RMSE={np.sqrt(mean_squared_error(y_true, y_pred)):.4f} "
          f"MAE={mean_absolute_error(y_true, y_pred):.4f} "
          f"R2={r2_score(y_true, y_pred):.4f}")


df, fc = load_and_clean()
df = add_time_features(df)
fc_time = fc + ["hour_sin", "hour_cos", "doy_sin", "doy_cos"]

for name, feat_cols, use_lags, use_rolling in [
        ("窗口252", fc, False, False),
        ("窗口252+时间", fc_time, False, False),
        ("窗口+时间+滞后", fc_time, True, False),
        ("窗口+时间+滞后+滚动", fc_time, True, True)]:
    Xf, y = build_features(df, feat_cols, use_lags=use_lags, use_rolling=use_rolling)
    tr, va, te, *_ = split_chronological(len(y))
    Xtr, Xte, ytr, yte = Xf[tr], Xf[te], y[tr], y[te]
    # LightGBM
    m = lgb.LGBMRegressor(n_estimators=800, learning_rate=0.05, num_leaves=63,
                          subsample=0.9, colsample_bytree=0.9, random_state=42,
                          verbose=-1)
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], callbacks=[lgb.early_stopping(50, verbose=False)])
    report(yte, m.predict(Xte, num_iteration=m.best_iteration_), "LGB " + name)

# XGBoost（最强特征集）
Xf, y = build_features(df, fc_time, use_lags=True, use_rolling=True)
tr, va, te, *_ = split_chronological(len(y))
Xtr, Xte, ytr, yte = Xf[tr], Xf[te], y[tr], y[te]
m = xgb.XGBRegressor(n_estimators=800, learning_rate=0.05, max_depth=8,
                     subsample=0.9, colsample_bytree=0.9, random_state=42,
                     early_stopping_rounds=50)
m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
report(yte, m.predict(Xte), "XGB 窗口+时间+滞后+滚动")
