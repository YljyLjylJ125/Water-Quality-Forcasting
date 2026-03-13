import os
import argparse
import time
import numpy as np
import pandas as pd
import scipy.io as sio
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = y_true != 0
    if np.sum(non_zero) == 0:
        return np.inf
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


def randomly_mask_series(series, missing_rate, seed=None):
    if seed is not None:
        np.random.seed(seed)
    series_masked = series.copy().astype(float)
    mask = np.random.rand(len(series)) < missing_rate
    series_masked[mask] = np.nan
    series_pd = pd.Series(series_masked).ffill().bfill()
    if series_pd.isna().any():
        series_pd = series_pd.fillna(series_pd.mean())
    return series_pd.values


def mask_lag_matrix(x, missing_rate, rng):
    if missing_rate <= 0:
        return x
    x_masked = x.copy().astype(float)
    miss = rng.rand(*x_masked.shape) < missing_rate
    x_masked[miss] = np.nan
    return x_masked


def make_lag_supervised(series, lags, horizon):
    x_list, y_list = [], []
    max_t = len(series) - horizon + 1
    for t in range(lags, max_t):
        x_list.append(series[t - lags:t])
        y_list.append(series[t:t + horizon])
    if len(x_list) == 0:
        return np.empty((0, lags)), np.empty((0, horizon))
    return np.asarray(x_list), np.asarray(y_list)


def iterative_block_forecast(model, history, n_steps, horizon, lags, missing_rate, rng):
    preds = []
    step_ids = []
    hist = list(history.astype(float))

    while len(preds) < n_steps:
        block_len = min(horizon, n_steps - len(preds))
        x = np.asarray(hist[-lags:]).reshape(1, -1)
        x = mask_lag_matrix(x, missing_rate=missing_rate, rng=rng)
        y_block = np.asarray(model.predict(x)).reshape(-1)  # [horizon]
        for s in range(block_len):
            y_hat = float(y_block[s])
            preds.append(y_hat)
            step_ids.append(s + 1)
            hist.append(y_hat)

    return np.asarray(preds), step_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed used for repeat runs.")
    args = parser.parse_args()

    start_time = time.time()

    dataset_candidates = ["/root/multi-water-quality/datasets/water_dataset.mat", "water_dataset.mat", "/root/autodl-tmp/water_dataset.mat"]
    dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if dataset_path is None:
        raise FileNotFoundError("Cannot find water_dataset.mat")

    data = sio.loadmat(dataset_path)
    Y_tr = data["Y_tr"]  # (37, 423)
    Y_te = data["Y_te"]  # (37, 282)

    n_points = 37
    n_test_days = 282
    lag_window = 12
    horizons = [1, 3, 5]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    all_results = []
    prediction_rows = []

    total_combinations = len(horizons) * len(missing_rates) * 3
    pbar_outer = tqdm(total=total_combinations, desc="overall", position=0, leave=True)

    for horizon in horizons:
        print(f"\n{'=' * 70}")
        print(f"XGBoost | horizon={horizon}")
        print(f"{'=' * 70}")

        for miss_rate in missing_rates:
            repeat_maes, repeat_rmses, repeat_mapes = [], [], []

            for repeat in range(3):
                seed = args.seed_base + repeat
                site_metrics = []

                pbar_site = tqdm(
                    range(n_points),
                    desc=f"H{horizon} MR{miss_rate:.1f} Rep{repeat+1}",
                    position=1,
                    leave=False,
                    ncols=100,
                )

                for i in pbar_site:
                    y_train = Y_tr[i, :].astype(float)
                    y_test = Y_te[i, :n_test_days]

                    x_train, y_target = make_lag_supervised(y_train, lag_window, horizon)
                    if len(x_train) == 0:
                        pred_vals = np.full(n_test_days, float(y_train[-1]))
                        step_ids = [(k % horizon) + 1 for k in range(n_test_days)]
                    else:
                        rng = np.random.RandomState(seed + 1000 + i)
                        x_train = mask_lag_matrix(x_train, missing_rate=miss_rate, rng=rng)
                        base = XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=300,
                            max_depth=4,
                            learning_rate=0.05,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            random_state=seed,
                            n_jobs=1,
                        )
                        model = MultiOutputRegressor(base, n_jobs=1)
                        model.fit(x_train, y_target)
                        rng_roll = np.random.RandomState(seed + 2000 + i)
                        pred_vals, step_ids = iterative_block_forecast(
                            model=model,
                            history=y_train,
                            n_steps=n_test_days,
                            horizon=horizon,
                            lags=lag_window,
                            missing_rate=miss_rate,
                            rng=rng_roll,
                        )

                    mae = mean_absolute_error(y_test, pred_vals)
                    rmse = np.sqrt(mean_squared_error(y_test, pred_vals))
                    mape = mean_absolute_percentage_error(y_test, pred_vals)
                    site_metrics.append({"mae": mae, "rmse": rmse, "mape": mape})

                    for t_idx, (yt, yp) in enumerate(zip(y_test, pred_vals)):
                        prediction_rows.append(
                            {
                                "model": "XGBOOST",
                                "missing_rate": miss_rate,
                                "horizon": horizon,
                                "repeat": repeat,
                                "site": i,
                                "step": int(step_ids[t_idx]),
                                "time_idx": t_idx,
                                "y_true": float(yt),
                                "y_pred": float(yp),
                            }
                        )

                    pbar_site.set_postfix({"MAE": f"{mae:.4f}", "RMSE": f"{rmse:.4f}", "MAPE": f"{mape:.2f}%"})

                repeat_maes.append(float(np.mean([m["mae"] for m in site_metrics])))
                repeat_rmses.append(float(np.mean([m["rmse"] for m in site_metrics])))
                repeat_mapes.append(float(np.mean([m["mape"] for m in site_metrics])))
                pbar_outer.update(1)

            final_mae = float(np.mean(repeat_maes))
            final_rmse = float(np.mean(repeat_rmses))
            final_mape = float(np.mean(repeat_mapes))

            all_results.append(
                {
                    "missing_rate": miss_rate,
                    "horizon": horizon,
                    "mae": final_mae,
                    "rmse": final_rmse,
                    "mape": final_mape,
                }
            )
            print(
                f"OK H={horizon}, MR={miss_rate:.1f} -> "
                f"MAE: {final_mae:.4f}, RMSE: {final_rmse:.4f}, MAPE: {final_mape:.2f}%"
            )

    pd.DataFrame(all_results)[["missing_rate", "horizon", "mae", "rmse", "mape"]].to_csv(
        "results_XGBOOST_iterative_fulltest_avg3.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(prediction_rows).to_csv("predictions_XGBOOST_plot_ready.csv", index=False, encoding="utf-8-sig")

    total_time = time.time() - start_time
    print(f"\nDone. Total time: {total_time / 60:.1f} min")

