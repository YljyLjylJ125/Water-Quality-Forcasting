import scipy.io as sio
import numpy as np
import pandas as pd
import os
import argparse
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
import time

warnings.filterwarnings("ignore")

# =========================================================
# 工具函数
# =========================================================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.inf
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

def interpolate_missing(series, missing_rate, seed=None):
    """
    对序列随机加缺失，并用线性插�?+ 均值填充处理，返回完整序列�?
    """
    if seed is not None:
        np.random.seed(seed)
    s = series.astype(float).copy()
    mask = np.random.rand(len(s)) < missing_rate
    s[mask] = np.nan

    ser = pd.Series(s)
    # 中间缺失：线性插�?
    ser = ser.interpolate(method='linear')
    # 首尾仍为 NaN 的部分：用均值填�?
    ser = ser.fillna(ser.mean())
    return ser.values


def iterative_forecast_fast(train_series, n_test_days, horizon, order=(3, 0, 2)):
    """
    Fast iterative forecasting:
    - fit ARIMA once
    - forecast block by block
    - append predicted block to state without refit
    """
    try:
        res = ARIMA(train_series, order=order).fit()
    except Exception:
        pred_vals = np.full(n_test_days, float(train_series[-1]))
        step_ids = [(k % horizon) + 1 for k in range(n_test_days)]
        return pred_vals, step_ids

    preds = []
    step_ids = []
    while len(preds) < n_test_days:
        block_len = min(horizon, n_test_days - len(preds))
        try:
            pred_block = np.asarray(res.forecast(steps=block_len))
        except Exception:
            pred_block = np.full(block_len, float(train_series[-1] if len(preds) == 0 else preds[-1]))

        preds.extend(pred_block.tolist())
        step_ids.extend(list(range(1, block_len + 1)))

        # Update model state with predicted observations without expensive refit.
        try:
            res = res.append(pred_block, refit=False)
        except Exception:
            pass

    return np.asarray(preds[:n_test_days]), step_ids[:n_test_days]

# =========================================================
# 主程序：传统 ARIMA（必须输入完整序列）
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed used for repeat runs.")
    args = parser.parse_args()

    start_time = time.time()
    
    # 加载数据（请确保 water_dataset.mat 在当前目录）
    dataset_candidates = [
        "/root/multi-water-quality/datasets/water_dataset.mat",
        "water_dataset.mat",
        "/root/autodl-tmp/water_dataset.mat",
    ]
    dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if dataset_path is None:
        raise FileNotFoundError("Cannot find water_dataset.mat")
    data = sio.loadmat(dataset_path)
    Y_tr = data['Y_tr']  # shape: (37, 423)
    Y_te = data['Y_te']  # shape: (37, 282)

    n_points = 37
    n_train_days = 423
    n_test_days = 282
    horizons = [1, 3, 5]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = []
    prediction_rows = []

    print("Traditional ARIMA experiment started")

    for horizon in horizons:
        print(f"\n{'='*70}")
        print(f"🔸 Horizon = {horizon}")
        print(f"{'='*70}")
        
        for miss_rate in missing_rates:
            repeat_maes, repeat_rmses, repeat_mapes = [], [], []
            
            for repeat in range(3):
                seed = args.seed_base + repeat
                site_metrics = []
                
                for i in range(n_points):
                    Y_tr_ph = Y_tr[i, :]   # 训练序列
                    Y_te_ph = Y_te[i, :]   # 测试序列

                    # 插值处理缺�?�?得到完整训练序列
                    Y_train_interpolated = interpolate_missing(Y_tr_ph, miss_rate, seed=seed)


                    pred_vals, step_list = iterative_forecast_fast(
                        train_series=Y_train_interpolated,
                        n_test_days=n_test_days,
                        horizon=horizon,
                        order=(3, 0, 2),
                    )
                    true_vals = Y_te_ph[:n_test_days]

                    for t_idx, (yt, yp) in enumerate(zip(true_vals, pred_vals)):
                            prediction_rows.append({
                                "model": "ARIMA",
                                "missing_rate": miss_rate,
                                "horizon": horizon,
                                "repeat": repeat,
                                "site": i,
                                "step": int(step_list[t_idx]),
                                "time_idx": t_idx,
                                "y_true": float(yt),
                                "y_pred": float(yp),
                            })

                    # 计算指标
                    mae = mean_absolute_error(true_vals, pred_vals)
                    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
                    mape = mean_absolute_percentage_error(true_vals, pred_vals)
                    site_metrics.append({"mae": mae, "rmse": rmse, "mape": mape})

                # 计算�?repeat �?37 个站点的平均指标
                avg_mae = np.mean([m["mae"] for m in site_metrics])
                avg_rmse = np.mean([m["rmse"] for m in site_metrics])
                avg_mape = np.mean([m["mape"] for m in site_metrics])
                
                repeat_maes.append(avg_mae)
                repeat_rmses.append(avg_rmse)
                repeat_mapes.append(avg_mape)

            # 三次重复取平�?
            final_mae = np.mean(repeat_maes)
            final_rmse = np.mean(repeat_rmses)
            final_mape = np.mean(repeat_mapes)

            all_results.append({
                "missing_rate": miss_rate,
                "horizon": horizon,
                "mae": final_mae,
                "rmse": final_rmse,
                "mape": final_mape
            })

            print(f"�?MR={miss_rate:.1f} �?MAE: {final_mae:.5f}, RMSE: {final_rmse:.5f}, MAPE: {final_mape:.3f}%")

    # 保存结果
    df = pd.DataFrame(all_results)
    df = df[["missing_rate", "horizon", "mae", "rmse", "mape"]]
    output_file = "arima_traditional_avg3.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    pd.DataFrame(prediction_rows).to_csv("predictions_ARIMA_plot_ready.csv", index=False, encoding="utf-8-sig")

    total_time = time.time() - start_time
    print(f"\n🎉 传统 ARIMA 实验完成！耗时 {total_time/60:.1f} 分钟")
    print(f"📊 结果已保存至: {output_file}")

