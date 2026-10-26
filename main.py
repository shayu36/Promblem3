"""任务 3：时间序列预测 —— 过去 2 小时预测下一时间点室外温度（OT）。

运行： python main.py
输出：控制台打印各模型指标，results/ 下保存可视化结果与指标。
"""
import json
import os

import numpy as np
import torch

from config import RESULTS_DIR, TARGET_COL, WINDOW_SIZE
from src.data_utils import (load_and_clean, build_windows, split_chronological,
                            scale_data)
from src.models import build_model
from src.train import train_model
from src.evaluate import (regression_metrics, persistence_baseline,
                          ridge_baseline, plot_loss, plot_prediction, plot_scatter)

os.makedirs(RESULTS_DIR, exist_ok=True)


def run_baselines(y_train_raw, y_test_raw, Xtr, Xte):
    """在原始量纲上评估两个基线（返回 raw 预测）。"""
    res = {}

    # 1) 持久化基线：用最后一个观测值外推一步（预测下一时刻温度不变）
    gt, pred = persistence_baseline(y_train_raw[-1:], y_test_raw)
    res["Persistence"] = regression_metrics(gt, pred)

    # 2) 岭回归基线：标准化特征上训练，原始量纲目标，直接输出原始量纲预测
    gt, pred = ridge_baseline(Xtr, y_train_raw, Xte, y_test_raw)
    res["Ridge"] = regression_metrics(gt, pred)

    return res


def run_recurrent(cell, Xtr, ytr, Xva, yva, Xte, mu, sd):
    """训练 LSTM/GRU，返回原始量纲下的测试集预测。"""
    print(f"\n=== 训练 {cell.upper()} ===")
    input_dim = Xtr.shape[-1]
    model = build_model(input_dim, cell=cell)
    model, history = train_model(model, Xtr, ytr, Xva, yva)

    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(Xte)).numpy()

    plot_loss(history, os.path.join(RESULTS_DIR, f"loss_{cell}.png"))
    return pred_scaled * sd + mu          # 还原到原始量纲


def main():
    df, feature_cols = load_and_clean()
    target_idx = feature_cols.index(TARGET_COL)

    # ---- 滑动窗口切分 ----
    X, y = build_windows(df, feature_cols, WINDOW_SIZE)
    print(f"[data] 滑动窗口 {WINDOW_SIZE} 步 -> 输入 {X.shape}, 目标 {y.shape}")

    # ---- 时序划分 ----
    tr_idx, va_idx, te_idx, n_tr, n_va, n_te = split_chronological(len(y))
    print(f"[data] 划分 train/val/test = {n_tr}/{n_va}/{n_te}")

    y_train_raw = y[tr_idx]          # 原始量纲，用于基线评估
    y_test_raw = y[te_idx]

    # ---- 标准化（仅在训练集上拟合，特征与目标同时标准化）----
    Xtr, Xva, Xte, ytr, yva, yte, mu, sd = scale_data(
        X[tr_idx], X[va_idx], X[te_idx], y[tr_idx], y[va_idx], y[te_idx],
        target_idx=target_idx)
    print(f"[data] 目标 OT 训练集均值 {mu:.2f}、标准差 {sd:.2f}")

    metrics = {}

    # ---- 基线（原始量纲）----
    metrics.update(run_baselines(y_train_raw, y_test_raw, Xtr, Xte))

    # ---- 深度学习模型 ----
    preds = {}
    for cell in ["lstm", "gru"]:
        preds[cell] = run_recurrent(cell, Xtr, ytr, Xva, yva, Xte, mu, sd)
        metrics[cell.upper()] = regression_metrics(yte * sd + mu, preds[cell])

    # ---- 汇总与可视化 ----
    print("\n=== 测试集评估结果（原始量纲）===")
    for name, m in metrics.items():
        print(f"  {name:12s} RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE(%)']:.2f}%")

    # 用最优模型（GRU）的预测画图
    y_true_raw = yte * sd + mu
    plot_prediction(y_true_raw, preds["gru"], n=240,
                    path=os.path.join(RESULTS_DIR, "prediction.png"))
    plot_scatter(y_true_raw, preds["gru"], path=os.path.join(RESULTS_DIR, "scatter.png"))

    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 {RESULTS_DIR}")


if __name__ == "__main__":
    main()
