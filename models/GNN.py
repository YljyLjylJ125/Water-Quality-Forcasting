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

# -------------------------
# Device
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

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

# -------------------------
# Temporal GNN (T-GCN style): GraphConv (spatial) + GRU (temporal)
# -------------------------
class AdaptiveAdjacency(nn.Module):
    """
    Learn an adaptive adjacency matrix A in R^{N x N} when no physical graph is provided.
    A = softmax(relu(E1 @ E2)) row-normalized.
    """
    def __init__(self, num_nodes, embed_dim=32):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(embed_dim, num_nodes) * 0.1)

    def forward(self):
        A = torch.relu(self.E1 @ self.E2)     # [N, N]
        A = torch.softmax(A, dim=1)           # row-normalized
        return A

class GraphConv(nn.Module):
    """
    Simple 1-hop graph convolution:
      x: [B, N, Fin] -> [B, N, Fout]
      x_agg = A @ x
    """
    def __init__(self, fin, fout, num_nodes, embed_dim=32, bias=True, use_adaptive=True):
        super().__init__()
        self.use_adaptive = use_adaptive
        self.num_nodes = num_nodes
        self.lin = nn.Linear(fin, fout, bias=bias)
        if use_adaptive:
            self.adpA = AdaptiveAdjacency(num_nodes, embed_dim=embed_dim)

    def forward(self, x, A_fixed=None):
        # x: [B, N, Fin]
        if self.use_adaptive:
            A_adp = self.adpA()  # [N, N]
            if A_fixed is None:
                A = A_adp
            else:
                A = 0.5 * A_fixed + 0.5 * A_adp
        else:
            if A_fixed is None:
                A = torch.eye(self.num_nodes, device=x.device)
            else:
                A = A_fixed

        # aggregate neighbors: [B, N, Fin] <- [N, N] @ [B, N, Fin]
        x_agg = torch.einsum('nm,bmf->bnf', A, x)
        return self.lin(x_agg)

class MultiSiteMaskedTGCN(nn.Module):
    """
    Input:  x [B, L, N, D]  (D=24 in your pipeline)
    Output: y [B, N]
    Steps:
      per t: GraphConv across nodes
      then:  GRU along time per node (shared weights)
      then:  Linear -> scalar per node
    """
    def __init__(
        self,
        N,
        input_dim_per_site=24,
        gnn_hidden=128,
        rnn_hidden=256,
        output_dim=1,
        dropout=0.1,
        embed_dim=32,
        A_fixed=None
    ):
        super().__init__()
        self.N = N
        self.input_dim_per_site = input_dim_per_site
        self.output_dim = output_dim
        self.A_fixed = A_fixed  # torch.Tensor [N, N] or None

        self.gconv = GraphConv(
            fin=input_dim_per_site,
            fout=gnn_hidden,
            num_nodes=N,
            embed_dim=embed_dim,
            use_adaptive=True
        )
        self.norm = nn.LayerNorm(gnn_hidden)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        self.gru = nn.GRU(input_size=gnn_hidden, hidden_size=rnn_hidden, batch_first=True)
        self.out = nn.Linear(rnn_hidden, output_dim)

    def forward(self, x):
        # x: [B, L, N, D]
        B, L, N, D = x.shape
        assert N == self.N and D == self.input_dim_per_site, \
            f"Got x shape {x.shape}, expected N={self.N}, D={self.input_dim_per_site}"

        A_fixed = None
        if self.A_fixed is not None:
            A_fixed = self.A_fixed.to(x.device)

        hs = []
        for t in range(L):
            xt = x[:, t, :, :]                    # [B, N, D]
            ht = self.gconv(xt, A_fixed=A_fixed)  # [B, N, gnn_hidden]
            ht = self.norm(ht)
            ht = self.act(ht)
            ht = self.drop(ht)
            hs.append(ht)

        H = torch.stack(hs, dim=1)                # [B, L, N, gnn_hidden]
        H = H.permute(0, 2, 1, 3).contiguous()    # [B, N, L, gnn_hidden]
        H = H.view(B * N, L, -1)                  # [B*N, L, gnn_hidden]
        _, h_last = self.gru(H)                   # [1, B*N, rnn_hidden]
        h_last = h_last[-1]                       # [B*N, rnn_hidden]

        y = self.out(h_last).view(B, N, self.output_dim)  # [B, N, H]
        return y

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
    X_tr_cell = data['X_tr']
    Y_tr = data['Y_tr']
    X_te_cell = data['X_te']
    Y_te = data['Y_te']

    def stack_X(X_cell):
        days = X_cell.shape[1]
        return np.stack([X_cell[0, t] for t in range(days)], axis=0)  # [T, N, 11]

    X_tr_all = stack_X(X_tr_cell)
    X_te_all = stack_X(X_te_cell)
    X_all = np.concatenate([X_tr_all, X_te_all], axis=0)  # [705, 37, 11]
    Y_all = np.concatenate([Y_tr.T, Y_te.T], axis=0)      # [705, 37]

    T_total, N, d_aux = X_all.shape
    print(f"Total T={T_total}, Sites N={N}, AuxDim={d_aux}")

    # -------------------------
    # Experiments (same as your current fairness setting)
    # -------------------------
    window = 12
    horizons = [1, 3, 5]
    missing_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    all_results = []
    prediction_rows = []

    for horizon in horizons:
        for miss_rate in missing_rates:
            print(f"{'='*60}")
            print(f"GNN(T-GCN) | missing_rate={miss_rate} | horizon={horizon} | repeats=3")
            print(f"{'='*60}")

            repeat_maes, repeat_rmses, repeat_mapes = [], [], []

            for repeat in range(3):
                seed = args.seed_base + repeat
                print(f"  �?�?{repeat+1}/3 次实�?(seed={seed})")

                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                # === 1. 标准化（每站点独立，仅用训练�?fit�?==
                scalers_X = [StandardScaler() for _ in range(N)]
                scalers_Y = [StandardScaler() for _ in range(N)]
                X_train_raw = X_all[:423]
                Y_train_raw = Y_all[:423]

                X_all_std = np.zeros_like(X_all)
                Y_all_std = np.zeros_like(Y_all)

                for i in range(N):
                    X_all_std[:423, i, :] = scalers_X[i].fit_transform(X_train_raw[:, i, :])
                    X_all_std[423:, i, :] = scalers_X[i].transform(X_all[423:, i, :])

                    Y_all_std[:423, i] = scalers_Y[i].fit_transform(Y_train_raw[:, i:i+1]).flatten()
                    Y_all_std[423:, i] = scalers_Y[i].transform(Y_all[423:, i:i+1]).flatten()

                # === 2. 构造序列：aux + hist pH ===
                T_valid = T_total - window - horizon + 1
                if T_valid <= 0:
                    print(f"⚠️ 跳过 horizon={horizon}：T_valid={T_valid} <= 0")
                    break

                X_seq_list, Y_target_list = [], []
                np.random.seed(seed)  # keep consistent with your original code

                for t in range(T_valid):
                    aux_seq = X_all_std[t:t+window]             # [12, N, 11]
                    ph_seq  = Y_all_std[t:t+window]             # [12, N]
                    input_seq = np.concatenate([aux_seq, ph_seq[..., None]], axis=-1)  # [12, N, 12]
                    X_seq_list.append(input_seq)
                    y_multi = Y_all_std[t+window:t+window+horizon]  # [H, N]
                    Y_target_list.append(y_multi.T)                  # [N, H]

                X_seq = np.array(X_seq_list)        # [T_valid, 12, N, 12]
                Y_targets = np.array(Y_target_list) # [T_valid, N, H]

                # === 3. 添加缺失（仅输入中的 pH 通道�?==
                X_with_missing = apply_missing_to_ph_in_input(
                    X_seq, missing_rate=miss_rate, ph_channel_idx=-1, seed=seed
                )  # [T_valid, 12, N, 24]

                # quick check
                ph_mask = X_with_missing[..., -1]
                print("    实际缺失�?按mask统计) =", float(1.0 - ph_mask.mean()))

                # === 4. 时间划分：训练期最�?2天为验证（保持原口径�?==
                train_end = 423 - window - horizon + 1
                val_size = 82
                if train_end <= val_size:
                    raise ValueError(f"Not enough training data for horizon={horizon}")

                T_train = train_end - val_size
                T_val = train_end

                X_train = X_with_missing[:T_train]
                X_val   = X_with_missing[T_train:T_val]

                Y_train = Y_targets[:T_train]
                Y_val   = Y_targets[T_train:T_val]

                X_train_t = torch.from_numpy(X_train).float().to(device)
                X_val_t   = torch.from_numpy(X_val).float().to(device)

                Y_train_t = torch.from_numpy(Y_train).float().to(device)
                Y_val_t   = torch.from_numpy(Y_val).float().to(device)

                # === 5. 模型训练 ===
                model = MultiSiteMaskedTGCN(
                    N=N, input_dim_per_site=24, gnn_hidden=128, rnn_hidden=256,
                    output_dim=horizon, dropout=0.1, embed_dim=32, A_fixed=None
                ).to(device)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                best_val_loss = float('inf')
                patience, trigger = 15, 0
                ckpt_path = f"tmp_best_gnn_h{horizon}_mr{miss_rate}_rep{repeat}.pth"

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
                        val_loss = criterion(val_pred, Y_val_t).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        trigger = 0
                        torch.save(model.state_dict(), ckpt_path)
                    else:
                        trigger += 1
                        if trigger >= patience:
                            break

                # === Load best ===
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                if os.path.exists(ckpt_path):
                    os.remove(ckpt_path)
                model.eval()

                # ============================================================
                # A) 第一层对 aux / pH / pH-mask 的权重范�?
                # ============================================================
                with torch.no_grad():
                    W = model.gconv.lin.weight.detach().cpu().numpy()  # [gnn_hidden, 24]
                w_aux = np.linalg.norm(W[:, 0:11])
                w_ph  = np.linalg.norm(W[:, 11])
                w_mph = np.linalg.norm(W[:, 23])  # mask for pH channel
                print(f"    [Diag-A] ||W_aux(0:11)||={w_aux:.6f}, ||W_ph(11)||={w_ph:.6f}, ||W_mask_ph(23)||={w_mph:.6f}")

                # ============================================================
                # 6) 测试评估（当�?miss_rate 下的 X_test_t�?
                # ============================================================
                # iterative rolling prediction over full test period
                y_roll_std = Y_all_std.copy()
                pred_blocks_std = []
                true_blocks_std = []
                step_blocks = []
                rng = np.random.RandomState(seed + 10000)

                with torch.no_grad():
                    for t_start in range(423, T_total, horizon):
                        block_len = min(horizon, T_total - t_start)
                        aux_seq = X_all_std[t_start-window:t_start]          # [window,N,11]
                        ph_seq = y_roll_std[t_start-window:t_start]          # [window,N]
                        input_seq = np.concatenate([aux_seq, ph_seq[..., None]], axis=-1)[None, ...]  # [1,window,N,12]

                        miss_seed = int(rng.randint(0, 10**9))
                        x_in = apply_missing_to_ph_in_input(
                            input_seq, missing_rate=miss_rate, ph_channel_idx=-1, seed=miss_seed
                        )  # [1,window,N,24]
                        x_in_t = torch.from_numpy(x_in).float().to(device)
                        pred_block_std = model(x_in_t).cpu().numpy()[0]  # [N,H]

                        y_roll_std[t_start:t_start+block_len] = pred_block_std[:, :block_len].T
                        pred_blocks_std.append(pred_block_std[:, :block_len].T)      # [block_len,N]
                        true_blocks_std.append(Y_all_std[t_start:t_start+block_len])  # [block_len,N]
                        step_blocks.append(np.arange(1, block_len + 1, dtype=int))

                pred_all_std = np.concatenate(pred_blocks_std, axis=0)  # [282,N]
                true_all_std = np.concatenate(true_blocks_std, axis=0)

                rmse_list, mae_list, mape_list = [], [], []
                for i_site in range(N):
                    pred_site = scalers_Y[i_site].inverse_transform(pred_all_std[:, i_site:i_site+1]).reshape(-1)
                    true_site = scalers_Y[i_site].inverse_transform(true_all_std[:, i_site:i_site+1]).reshape(-1)
                    rmse_list.append(np.sqrt(mean_squared_error(true_site, pred_site)))
                    mae_list.append(mean_absolute_error(true_site, pred_site))
                    mape_list.append(mean_absolute_percentage_error(true_site, pred_site))

                avg_mae = float(np.mean(mae_list))
                avg_rmse = float(np.mean(rmse_list))
                avg_mape = float(np.mean(mape_list))

                offset = 0
                for blk_steps in step_blocks:
                    block_len = len(blk_steps)
                    pred_blk = pred_all_std[offset:offset+block_len]
                    true_blk = true_all_std[offset:offset+block_len]
                    for row_i, st in enumerate(blk_steps):
                        for i_site in range(N):
                            yp = scalers_Y[i_site].inverse_transform(pred_blk[row_i, i_site:i_site+1].reshape(1, 1))[0, 0]
                            yt = scalers_Y[i_site].inverse_transform(true_blk[row_i, i_site:i_site+1].reshape(1, 1))[0, 0]
                            prediction_rows.append({
                                "model": "GNN",
                                "missing_rate": miss_rate,
                                "horizon": horizon,
                                "repeat": repeat,
                                "site": i_site,
                                "step": int(st),
                                "time_idx": int(offset + row_i),
                                "y_true": float(yt),
                                "y_pred": float(yp),
                            })
                    offset += block_len
                repeat_maes.append(avg_mae)
                repeat_rmses.append(avg_rmse)
                repeat_mapes.append(avg_mape)
                print(f"    �?(miss={miss_rate}) MAE: {avg_mae:.6f}, RMSE: {avg_rmse:.6f}, MAPE: {avg_mape:.4f}%")


            # === 3次平�?===
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

    # Save
    df = pd.DataFrame(all_results)
    df = df[['missing_rate', 'horizon', 'label_missing', 'mae', 'rmse', 'mape']]
    out_csv = "results_GNN_multisite_fair_avg3.csv"
    df.to_csv(out_csv, index=False)
    pd.DataFrame(prediction_rows).to_csv("predictions_GNN_plot_ready.csv", index=False)
    print(f"🎉 完成！结果保存至 {out_csv}")
    print(df)


