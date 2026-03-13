# lstm.py
import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.inf
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def apply_missing_to_ph_in_input(X_input, missing_rate, ph_channel_idx=-1, seed=None):
    """X_input: [T, window, D] where D = aux_dim + 1 (last dim is historical pH)"""
    if seed is not None:
        np.random.seed(seed)
    X_val = X_input.copy()
    mask = np.ones_like(X_input)
    T, L, D = X_input.shape
    missing = np.random.rand(T, L) < missing_rate
    X_val[:, :, ph_channel_idx][missing] = 0.0
    mask[:, :, ph_channel_idx][missing] = 0.0
    return np.concatenate([X_val, mask], axis=-1)  # [T, L, 2*D]

class SingleSiteLSTM(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        # x: [B, L, input_dim]
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])  # [B, output_dim]
        return pred

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
    X_tr_cell = data['X_tr']  # [1, 423]
    Y_tr = data['Y_tr']      # [37, 423]
    X_te_cell = data['X_te']  # [1, 282]
    Y_te = data['Y_te']      # [37, 282]

    def stack_X(X_cell):
        days = X_cell.shape[1]
        return np.stack([X_cell[0, t] for t in range(days)], axis=0)  # [T, N, 11]

    X_tr_all = stack_X(X_tr_cell)  # [423, 37, 11]
    X_te_all = stack_X(X_te_cell)  # [282, 37, 11]
    X_all = np.concatenate([X_tr_all, X_te_all], axis=0)  # [705, 37, 11]
    Y_all = np.concatenate([Y_tr.T, Y_te.T], axis=0)     # [705, 37]

    T_total, N, d_aux = X_all.shape
    print(f"Total T={T_total}, Sites N={N}")

    window = 12
    horizons = [1, 3, 5]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = []
    prediction_rows = []

    for horizon in horizons:
        for miss_rate in missing_rates:
            print(f"{'='*60}")
            print(f"LSTM | missing_rate={miss_rate} | horizon={horizon} | repeats=3")
            print(f"{'='*60}")
            repeat_maes, repeat_rmses, repeat_mapes = [], [], []
            for repeat in range(3):
                seed = args.seed_base + repeat
                site_metrics = []
                for i in range(N):  # 遍历每个站点
                    # === 1. 标准化（仅当前站点）===
                    scaler_X = StandardScaler()
                    scaler_Y = StandardScaler()

                    X_train_raw = X_all[:423, i, :]  # [423, 11]
                    Y_train_raw = Y_all[:423, i]     # [423,]

                    X_all_std_i = np.zeros((T_total, d_aux))
                    Y_all_std_i = np.zeros(T_total)

                    X_all_std_i[:423] = scaler_X.fit_transform(X_train_raw)
                    X_all_std_i[423:] = scaler_X.transform(X_all[423:, i, :])
                    Y_all_std_i[:423] = scaler_Y.fit_transform(Y_train_raw.reshape(-1, 1)).flatten()
                    Y_all_std_i[423:] = scaler_Y.transform(Y_all[423:, i].reshape(-1, 1)).flatten()

                    # === 2. 构造序列（仅当前站点）===
                    T_valid = T_total - window - horizon + 1
                    X_seq_list = []
                    Y_target_list = []
                    np.random.seed(seed)
                    for t in range(T_valid):
                        aux_seq = X_all_std_i[t:t+window]          # [12, 11]
                        ph_seq = Y_all_std_i[t:t+window]           # [12,]
                        input_seq = np.concatenate([aux_seq, ph_seq[:, None]], axis=-1)  # [12, 12]
                        X_seq_list.append(input_seq)
                        Y_target_list.append(Y_all_std_i[t + window:t + window + horizon])

                    X_seq = np.array(X_seq_list)      # [T_valid, 12, 12]
                    Y_targets = np.array(Y_target_list)  # [T_valid, H]

                    # === 3. 添加缺失（仅历史 pH�?==
                    # Train/val on clean inputs; only test-time context is corrupted by miss_rate.
                    X_with_missing = apply_missing_to_ph_in_input(
                        X_seq, missing_rate=0.0, ph_channel_idx=-1, seed=seed
                    )  # [T_valid, 12, 24]

                    # === 4. 时间划分 ===
                    train_end = 423 - window - horizon + 1
                    val_size = 82
                    if train_end <= val_size:
                        raise ValueError(f"Not enough training data for horizon={horizon}")
                    T_train = train_end - val_size
                    T_val = train_end

                    X_train = X_with_missing[:T_train]
                    X_val = X_with_missing[T_train:T_val]
                    X_test = X_with_missing[T_val:]
                    Y_train = Y_targets[:T_train]
                    Y_val = Y_targets[T_train:T_val]
                    Y_test = Y_targets[T_val:]

                    # === 5. �?tensor ===
                    X_train_t = torch.from_numpy(X_train).float().to(device)
                    X_val_t = torch.from_numpy(X_val).float().to(device)
                    X_test_t = torch.from_numpy(X_test).float().to(device)
                    Y_train_t = torch.from_numpy(Y_train).float().to(device)
                    Y_val_t = torch.from_numpy(Y_val).float().to(device)
                    Y_test_t = torch.from_numpy(Y_test).float().to(device)

                    # === 6. 训练模型（每个站点独立）===
                    model = SingleSiteLSTM(input_dim=24, hidden_dim=64, output_dim=horizon).to(device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=1e-3)
                    best_val_loss = float('inf')
                    patience, trigger = 15, 0

                    for epoch in range(200):
                        model.train()
                        train_pred = model(X_train_t)
                        loss = criterion(train_pred, Y_train_t)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            val_pred = model(X_val_t)
                            val_loss = criterion(val_pred, Y_val_t)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            trigger = 0
                            torch.save(model.state_dict(), f'tmp_best_lstm_site{i}_rep{repeat}.pth')
                        else:
                            trigger += 1
                            if trigger >= patience:
                                break

                    # === 7. 测试评估 ===
                    model.load_state_dict(torch.load(f'tmp_best_lstm_site{i}_rep{repeat}.pth', map_location=device))
                    os.remove(f'tmp_best_lstm_site{i}_rep{repeat}.pth')
                    model.eval()
                    # iterative rolling prediction over full test span
                    y_roll_std_i = Y_all_std_i.copy()
                    pred_list_std = []
                    true_list_std = []
                    step_list = []
                    rng = np.random.RandomState(seed + 10000 + i)

                    with torch.no_grad():
                        for t_start in range(423, T_total, horizon):
                            block_len = min(horizon, T_total - t_start)
                            aux_seq = X_all_std_i[t_start-window:t_start]
                            ph_seq = y_roll_std_i[t_start-window:t_start]
                            input_seq = np.concatenate([aux_seq, ph_seq[:, None]], axis=-1)[None, ...]  # [1,window,12]

                            miss_seed = int(rng.randint(0, 10**9))
                            x_in = apply_missing_to_ph_in_input(
                                input_seq, missing_rate=miss_rate, ph_channel_idx=-1, seed=miss_seed
                            )  # [1,window,24]
                            x_in_t = torch.from_numpy(x_in).float().to(device)

                            pred_block_std = model(x_in_t).cpu().numpy().reshape(-1)  # [H]
                            y_roll_std_i[t_start:t_start+block_len] = pred_block_std[:block_len]

                            pred_list_std.append(pred_block_std[:block_len])
                            true_list_std.append(Y_all_std_i[t_start:t_start+block_len])
                            step_list.append(np.arange(1, block_len + 1, dtype=int))

                    pred_i_std = np.concatenate(pred_list_std)
                    true_i_std = np.concatenate(true_list_std)
                    pred_i = scaler_Y.inverse_transform(pred_i_std.reshape(-1, 1)).reshape(-1)
                    true_i = scaler_Y.inverse_transform(true_i_std.reshape(-1, 1)).reshape(-1)

                    pos = 0
                    for blk_steps in step_list:
                        for row_i, st in enumerate(blk_steps):
                            prediction_rows.append({
                                "model": "LSTM",
                                "missing_rate": miss_rate,
                                "horizon": horizon,
                                "repeat": repeat,
                                "site": i,
                                "step": int(st),
                                "time_idx": int(pos + row_i),
                                "y_true": float(true_i[pos + row_i]),
                                "y_pred": float(pred_i[pos + row_i]),
                            })
                        pos += len(blk_steps)

                    rmse = np.sqrt(mean_squared_error(true_i, pred_i))
                    mae = mean_absolute_error(true_i, pred_i)
                    mape = mean_absolute_percentage_error(true_i, pred_i)
                    site_metrics.append({"mae": mae, "rmse": rmse, "mape": mape})

                # 平均所有站�?
                avg_mae = np.mean([m["mae"] for m in site_metrics])
                avg_rmse = np.mean([m["rmse"] for m in site_metrics])
                avg_mape = np.mean([m["mape"] for m in site_metrics])
                repeat_maes.append(avg_mae)
                repeat_rmses.append(avg_rmse)
                repeat_mapes.append(avg_mape)
                print(f" �?�?{repeat+1}/3 次实�?�?MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, MAPE: {avg_mape:.2f}%")

            # 3 次重复取平均
            final_mae = np.mean(repeat_maes)
            final_rmse = np.mean(repeat_rmses)
            final_mape = np.mean(repeat_mapes)
            all_results.append({
                'missing_rate': miss_rate,
                'horizon': horizon,
                'label_missing': False,
                'mae': final_mae,
                'rmse': final_rmse,
                'mape': final_mape
            })
            print(f"�?平均结果 �?MAE: {final_mae:.4f}, RMSE: {final_rmse:.4f}, MAPE: {final_mape:.2f}%")

    # 保存结果
    df = pd.DataFrame(all_results)
    df = df[['missing_rate', 'horizon', 'label_missing', 'mae', 'rmse', 'mape']]
    df.to_csv("results_lstm_per_site_avg3.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv("predictions_LSTM_plot_ready.csv", index=False)
    print("🎉 完成！结果保存至 results_lstm_per_site_avg3.csv")


