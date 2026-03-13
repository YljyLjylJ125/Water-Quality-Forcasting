import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

# -------------------------
# Metrics & Missing
# -------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def apply_missing_to_ph_in_input(X_input, missing_rate, ph_channel_idx=-1, seed=None):
    """
    X_input: [T, window, N, D] where D = aux_dim + 1 (last dim is historical pH)
    Only mask the pH channel in input sequences.
    Return concatenated [X, mask] => [T, window, N, 2*D]
    """
    if seed is not None:
        np.random.seed(seed)
    X_val = X_input.copy()
    mask = np.ones_like(X_input)

    T, L, N, D = X_input.shape
    missing = np.random.rand(T, L, N) < missing_rate
    X_val[:, :, :, ph_channel_idx][missing] = 0.0
    mask[:, :, :, ph_channel_idx][missing] = 0.0

    return np.concatenate([X_val, mask], axis=-1)  # [T, L, N, 2*D]

def flatten_X(X):
    """
    X: [B, window, N, D] -> [B, window*N*D]
    """
    return X.reshape(X.shape[0], -1)

def eval_preds_inverse_scaling(test_pred_std, Y_test_std, scalers_Y):
    """
    test_pred_std: [B, N, H] in standardized space
    Y_test_std:    [B, N, H] in standardized space
    Returns avg metrics across sites.
    """
    N = test_pred_std.shape[1]
    rmse_list, mae_list, mape_list = [], [], []

    for i in range(N):
        pred_i = scalers_Y[i].inverse_transform(test_pred_std[:, i, :].reshape(-1, 1)).reshape(-1)
        true_i = scalers_Y[i].inverse_transform(Y_test_std[:, i, :].reshape(-1, 1)).reshape(-1)

        rmse = np.sqrt(mean_squared_error(true_i, pred_i))
        mae = mean_absolute_error(true_i, pred_i)
        mape = mean_absolute_percentage_error(true_i, pred_i)

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)

    return float(np.mean(mae_list)), float(np.mean(rmse_list)), float(np.mean(mape_list))

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-base", type=int, default=42, help="Base seed used for repeat runs.")
    args = parser.parse_args()

    dataset_candidates = [
        "/root/multi-water-quality/datasets/water_dataset.mat",
        "water_dataset.mat",
        "/root/autodl-tmp/water_dataset.mat",
    ]
    dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
    if dataset_path is None:
        raise FileNotFoundError("Cannot find water_dataset.mat")

    data = sio.loadmat(dataset_path)
    X_tr_cell = data['X_tr']   # [1, 423]
    Y_tr = data['Y_tr']        # [37, 423]
    X_te_cell = data['X_te']   # [1, 282]
    Y_te = data['Y_te']        # [37, 282]

    def stack_X(X_cell):
        days = X_cell.shape[1]
        return np.stack([X_cell[0, t] for t in range(days)], axis=0)  # [T, N, 11]

    X_tr_all = stack_X(X_tr_cell)                         # [423, 37, 11]
    X_te_all = stack_X(X_te_cell)                         # [282, 37, 11]
    X_all = np.concatenate([X_tr_all, X_te_all], axis=0)  # [705, 37, 11]
    Y_all = np.concatenate([Y_tr.T, Y_te.T], axis=0)      # [705, 37]

    T_total, N, d_aux = X_all.shape
    print(f"Total T={T_total}, Sites N={N}, AuxDim={d_aux}")

    # -------------------------
    # Experiments (keep same protocol)
    # -------------------------
    window = 12
    horizons = [1, 3, 5]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # KNN candidate K list (use val to pick best K per setting)
    K_candidates = [1, 3, 5, 7, 9, 11, 15]

    all_results = []
    prediction_rows = []

    for horizon in horizons:
        for miss_rate in missing_rates:
            print(f"{'='*60}")
            print(f"KNN | missing_rate={miss_rate} | horizon={horizon} | repeats=3")
            print(f"{'='*60}")

            repeat_maes, repeat_rmses, repeat_mapes = [], [], []

            for repeat in range(3):
                seed = args.seed_base + repeat
                print(f"  �?�?{repeat+1}/3 次实�?(seed={seed})")

                # === 1. 标准化（每个站点独立，仅用训练期 fit�?==
                scalers_X = [StandardScaler() for _ in range(N)]
                scalers_Y = [StandardScaler() for _ in range(N)]

                X_train_raw = X_all[:423]  # [423, N, 11]
                Y_train_raw = Y_all[:423]  # [423, N]

                X_all_std = np.zeros_like(X_all)
                Y_all_std = np.zeros_like(Y_all)

                for i in range(N):
                    X_all_std[:423, i, :] = scalers_X[i].fit_transform(X_train_raw[:, i, :])
                    X_all_std[423:, i, :] = scalers_X[i].transform(X_all[423:, i, :])

                    Y_all_std[:423, i] = scalers_Y[i].fit_transform(Y_train_raw[:, i:i+1]).flatten()
                    Y_all_std[423:, i] = scalers_Y[i].transform(Y_all[423:, i:i+1]).flatten()

                # === 2. 构造序列：输入包含 aux + hist pH ===
                T_valid = T_total - window - horizon + 1
                if T_valid <= 0:
                    print(f"⚠️ 跳过 horizon={horizon}：T_valid={T_valid} <= 0")
                    break

                X_seq_list = []
                Y_target_list = []

                np.random.seed(seed)  # 保持你的缺失随机性控制逻辑一�?

                for t in range(T_valid):
                    aux_seq = X_all_std[t:t+window]                  # [12, N, 11]
                    ph_seq  = Y_all_std[t:t+window]                  # [12, N]
                    input_seq = np.concatenate([aux_seq, ph_seq[..., None]], axis=-1)  # [12, N, 12]
                    X_seq_list.append(input_seq)
                    y_multi = Y_all_std[t+window:t+window+horizon]  # [H, N]
                    Y_target_list.append(y_multi.T)                  # [N, H]

                X_seq = np.array(X_seq_list)         # [T_valid, 12, N, 12]
                Y_targets = np.array(Y_target_list)  # [T_valid, N, H]

                # === 3. 添加缺失（仅输入中的 pH 通道�?==
                X_with_missing = apply_missing_to_ph_in_input(
                    X_seq, missing_rate=miss_rate, ph_channel_idx=-1, seed=seed
                )  # [T_valid, 12, N, 24]

                ph_mask = X_with_missing[..., -1]
                print("    实际缺失�?按mask统计) =", float(1.0 - ph_mask.mean()))

                # === 4. 时间划分：训练期最�?2天为验证 ===
                train_end = 423 - window - horizon + 1
                val_size = 82
                if train_end <= val_size:
                    raise ValueError(f"Not enough training data for horizon={horizon}")

                T_train = train_end - val_size
                T_val = train_end

                X_train = X_with_missing[:T_train]
                X_val   = X_with_missing[T_train:T_val]
                X_test  = X_with_missing[T_val:]

                Y_train = Y_targets[:T_train]
                Y_val   = Y_targets[T_train:T_val]
                Y_test  = Y_targets[T_val:]
                Y_train_flat = Y_train.reshape(Y_train.shape[0], -1)
                Y_val_flat = Y_val.reshape(Y_val.shape[0], -1)

                # ---- KNN needs 2D feature matrix ----
                X_train_f = flatten_X(X_train)  # [B_train, 12*37*24]
                X_val_f   = flatten_X(X_val)
                X_test_f  = flatten_X(X_test)

                # === 5. “训�?选择模型”：�?val 选择最�?K（公平替�?early-stopping�?==
                best_k = None
                best_val_mse = float("inf")
                best_model = None

                # 确保 K 不超过训练样本数
                max_k_allowed = X_train_f.shape[0]
                K_list = [k for k in K_candidates if k <= max_k_allowed]
                if len(K_list) == 0:
                    K_list = [min(1, max_k_allowed)]

                for k in K_list:
                    model = KNeighborsRegressor(
                        n_neighbors=k,
                        weights="distance",
                        algorithm="brute",   # 高维 + 样本不大，brute 更稳
                        metric="minkowski",
                        p=2,
                        n_jobs=-1
                    )
                    model.fit(X_train_f, Y_train_flat)
                    val_pred = model.predict(X_val_f)
                    val_mse = float(np.mean((val_pred - Y_val_flat) ** 2))

                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_k = k
                        best_model = model

                print(f"    选定 best_k={best_k} (val_mse={best_val_mse:.6f})")

                # === 6. 测试评估：迭代滚动预测覆盖全部测试集 ===
                y_roll_std = Y_all_std.copy()
                test_pred_blocks_std = []
                test_true_blocks_std = []
                test_step_blocks = []
                rng = np.random.RandomState(seed + 10000)

                for t_start in range(423, T_total, horizon):
                    block_len = min(horizon, T_total - t_start)
                    aux_seq = X_all_std[t_start-window:t_start]         # [window, N, 11]
                    ph_seq = y_roll_std[t_start-window:t_start]         # [window, N]
                    input_seq = np.concatenate([aux_seq, ph_seq[..., None]], axis=-1)[None, ...]  # [1,window,N,12]

                    # iterative stage also applies input-side missing process
                    miss_seed = int(rng.randint(0, 10**9))
                    x_in = apply_missing_to_ph_in_input(
                        input_seq, missing_rate=miss_rate, ph_channel_idx=-1, seed=miss_seed
                    )  # [1,window,N,24]

                    pred_block_std = best_model.predict(flatten_X(x_in)).reshape(N, horizon)  # [N,H]
                    y_roll_std[t_start:t_start+block_len] = pred_block_std[:, :block_len].T

                    test_pred_blocks_std.append(pred_block_std[:, :block_len].T)  # [block_len,N]
                    test_true_blocks_std.append(Y_all_std[t_start:t_start+block_len])  # [block_len,N]
                    test_step_blocks.append(np.arange(1, block_len + 1, dtype=int))

                test_pred_std_all = np.concatenate(test_pred_blocks_std, axis=0)  # [282,N]
                test_true_std_all = np.concatenate(test_true_blocks_std, axis=0)   # [282,N]

                rmse_list, mae_list, mape_list = [], [], []
                for i_site in range(N):
                    pred_site = scalers_Y[i_site].inverse_transform(
                        test_pred_std_all[:, i_site:i_site+1]
                    ).reshape(-1)
                    true_site = scalers_Y[i_site].inverse_transform(
                        test_true_std_all[:, i_site:i_site+1]
                    ).reshape(-1)

                    rmse_list.append(np.sqrt(mean_squared_error(true_site, pred_site)))
                    mae_list.append(mean_absolute_error(true_site, pred_site))
                    mape_list.append(mean_absolute_percentage_error(true_site, pred_site))

                avg_mae = float(np.mean(mae_list))
                avg_rmse = float(np.mean(rmse_list))
                avg_mape = float(np.mean(mape_list))

                offset = 0
                for blk_idx, steps in enumerate(test_step_blocks):
                    block_len = len(steps)
                    pred_blk = test_pred_std_all[offset:offset+block_len]  # [block_len,N]
                    true_blk = test_true_std_all[offset:offset+block_len]
                    for row_i in range(block_len):
                        for i_site in range(N):
                            yp = scalers_Y[i_site].inverse_transform(pred_blk[row_i, i_site:i_site+1].reshape(1, 1))[0, 0]
                            yt = scalers_Y[i_site].inverse_transform(true_blk[row_i, i_site:i_site+1].reshape(1, 1))[0, 0]
                            prediction_rows.append({
                                "model": "KNN",
                                "missing_rate": miss_rate,
                                "horizon": horizon,
                                "repeat": repeat,
                                "site": i_site,
                                "step": int(steps[row_i]),
                                "time_idx": int(offset + row_i),
                                "y_true": float(yt),
                                "y_pred": float(yp),
                            })
                    offset += block_len
                repeat_maes.append(avg_mae)
                repeat_rmses.append(avg_rmse)
                repeat_mapes.append(avg_mape)

                print(f"    �?MAE: {avg_mae:.6f}, RMSE: {avg_rmse:.6f}, MAPE: {avg_mape:.4f}%")

            # === 计算3次平�?===
            final_mae = float(np.mean(repeat_maes))
            final_rmse = float(np.mean(repeat_rmses))
            final_mape = float(np.mean(repeat_mapes))

            all_results.append({
                'missing_rate': miss_rate,
                'horizon': horizon,
                'label_missing': False,
                'mae': final_mae,
                'rmse': final_rmse,
                'mape': final_mape
            })

            print(f"�?平均结果 �?MAE: {final_mae:.6f}, RMSE: {final_rmse:.6f}, MAPE: {final_mape:.4f}%")

    # Save results with desired column order
    df = pd.DataFrame(all_results)
    df = df[['missing_rate', 'horizon', 'label_missing', 'mae', 'rmse', 'mape']]
    out_csv = "results_KNN_multisite_fair_avg3.csv"
    df.to_csv(out_csv, index=False)
    pd.DataFrame(prediction_rows).to_csv("predictions_KNN_plot_ready.csv", index=False)
    print(f"🎉 完成！结果保存至 {out_csv}")
    print(df)



