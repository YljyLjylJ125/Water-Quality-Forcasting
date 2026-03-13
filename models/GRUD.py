# GRUD.py
# �?完全对齐流程：Δt �?γ �?x_tilde = γ⊙x_t + (1-γ)⊙x_last
# �?无泄露：静态特�?聚类�?标准�?fit 全部只用训练�?�?23�?
# �?主任务：预测（forecasting），监督目标是未来点 i+window+horizon-1
# �?修复 pin_memory 报错：DataLoader 只喂 CPU tensors；训练循环里再搬�?GPU

import os
import argparse
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import GATConv

# ========================
# Config
# ========================
MAT_CANDIDATES = ["/root/multi-water-quality/datasets/water_dataset.mat", "water_dataset.mat", "/root/autodl-tmp/water_dataset.mat"]
MAT_PATH = next((p for p in MAT_CANDIDATES if os.path.exists(p)), MAT_CANDIDATES[0])
A_S_CANDIDATES = ["/root/multi-water-quality/datasets/adjacency_matrix.npy", "adjacency_matrix.npy", "/root/autodl-tmp/adjacency_matrix.npy"]
A_S_PATH = next((p for p in A_S_CANDIDATES if os.path.exists(p)), A_S_CANDIDATES[0])  # 空间/结构先验图（假定外生给定�?
TRAIN_DAYS = 423

WINDOW = 12
HORIZONS = [1, 3, 5]
MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
REPEATS = 3
KEEP_CHECKPOINTS = True

# 聚类构图（仅用训练段�?
K_PH = 4
K_AUX = 4
DTW_MAX_ITER = 20

# 模型超参
D_FEAT = 32
HIDDEN_SIZE = 64
MOE_DIM = 8
NUM_EXPERTS = 4
NUM_HEADS = 4

# 训练超参
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 200
PATIENCE = 15
VAL_SIZE = 82
VAL_EVERY = 1

SEED_BASE = 42
INPUT_MODES = ["full"]  # options: full, x_only, mask_delta_only
OBS_DROPOUT_RATE = 0.00
DELTA_CLIP_MAX = 32.0
DELTA_USE_LOG1P = True

# DataLoader
NUM_WORKERS = 0  # 你可改成 2/4，但先用0最�?

# ========================
# Device
# ========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = (device.type == "cuda")
print(f"使用设备: {device}, pin_memory={PIN_MEMORY}")

# ========================
# Metrics
# ========================
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# ========================
# MoE
# ========================
class MoELayer(nn.Module):
    def __init__(self, input_dim=11, moe_dim=8, num_experts=4, hidden=32):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, moe_dim)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):  # x: [B*N, 11]
        g = F.softmax(self.gate(x), dim=-1)                 # [B*N, E]
        outs = torch.stack([e(x) for e in self.experts], 1) # [B*N, E, moe_dim]
        y = torch.sum(g.unsqueeze(-1) * outs, dim=1)        # [B*N, moe_dim]
        return y


# ========================
# Three-Graph GAT
# ========================
class ThreeGraphGAT(nn.Module):
    def __init__(self, d_p=3, d_a=11, d_feat=32, num_heads=4):
        super().__init__()
        self.align_p = nn.Linear(d_p, d_feat)
        self.align_a = nn.Linear(d_a, d_feat)
        self.align_s = nn.Linear(d_p + d_a, d_feat)

        hidden_dim = d_feat // num_heads
        self.gat_s = GATConv(d_feat, hidden_dim, heads=num_heads, concat=True)
        self.gat_p = GATConv(d_feat, hidden_dim, heads=num_heads, concat=True)
        self.gat_a = GATConv(d_feat, hidden_dim, heads=num_heads, concat=True)

        self.fuse = nn.Sequential(
            nn.Linear(3 * d_feat, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, d_feat),
        )

    def forward(self, F_p_raw, F_a_raw, edge_index_s, edge_index_p, edge_index_a):
        f_s = self.align_s(torch.cat([F_p_raw, F_a_raw], dim=-1))
        f_p = self.align_p(F_p_raw)
        f_a = self.align_a(F_a_raw)

        h_s = F.elu(self.gat_s(f_s, edge_index_s))
        h_p = F.elu(self.gat_p(f_p, edge_index_p))
        h_a = F.elu(self.gat_a(f_a, edge_index_a))

        h = self.fuse(torch.cat([h_s, h_p, h_a], dim=-1))  # [N, d_feat]
        return h


# ========================
# GRU-D Forecasting Model
# 核心：x_tilde = γ⊙x_t + (1-γ)⊙x_last
# ========================
class GRUDModel(nn.Module):
    def __init__(self, N, d_feat=32, aux_dim=11, moe_dim=8,
                 lookback=12, hidden_size=64, num_experts=4, num_heads=4, output_dim=1,
                 input_mode="full", delta_clip_max=32.0, delta_use_log1p=True):
        super().__init__()
        self.N = N
        self.lookback = lookback
        self.aux_dim = aux_dim
        self.moe_dim = moe_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.total_input_dim = 1 + moe_dim  # pH + z_aux
        self.input_mode = input_mode
        self.delta_clip_max = delta_clip_max
        self.delta_use_log1p = delta_use_log1p

        self.static_gat = ThreeGraphGAT(d_p=3, d_a=aux_dim, d_feat=d_feat, num_heads=num_heads)

        self.h0_init = nn.Sequential(
            nn.Linear(d_feat, d_feat),
            nn.ReLU(),
            nn.Linear(d_feat, hidden_size),
        )

        self.moe = MoELayer(input_dim=aux_dim, moe_dim=moe_dim, num_experts=num_experts)

        # decay params (diagonal)
        self.W_decay = nn.Parameter(torch.randn(self.total_input_dim))
        self.b_decay = nn.Parameter(torch.randn(self.total_input_dim))

        self.gru_cell = nn.GRUCell(input_size=self.total_input_dim, hidden_size=hidden_size)
        self.predictor = nn.Linear(hidden_size, output_dim)

        # static info buffers
        self.register_buffer("F_p_raw", torch.zeros(0))
        self.register_buffer("F_a_raw", torch.zeros(0))
        self.edge_index_s = None
        self.edge_index_p = None
        self.edge_index_a = None

    def set_static_info(self, F_p_raw, F_a_raw, edge_index_s, edge_index_p, edge_index_a):
        self.F_p_raw = F_p_raw
        self.F_a_raw = F_a_raw
        self.edge_index_s = edge_index_s
        self.edge_index_p = edge_index_p
        self.edge_index_a = edge_index_a

    def forward(self, X_aux_window, Y_ph_window, mask_ph_window):
        """
        X_aux_window: [B, T, N, 11]
        Y_ph_window : [B, T, N]
        mask_ph_window: [B, T, N]  (1 observed, 0 missing)
        return: [B, N, H]  最后一�?hidden �?multi-step forecast
        """
        B = X_aux_window.size(0)
        dev = X_aux_window.device

        if self.F_p_raw.numel() == 0:
            raise RuntimeError("Please call set_static_info before forward().")

        # ---- static prior -> h0 ----
        h0_node = self.static_gat(self.F_p_raw, self.F_a_raw,
                                 self.edge_index_s, self.edge_index_p, self.edge_index_a)  # [N,d_feat]
        h0 = self.h0_init(h0_node)  # [N, hidden]
        h = h0.unsqueeze(0).expand(B, -1, -1).reshape(B * self.N, self.hidden_size)  # [B*N, hidden]

        # ---- state: x_last and tau_last ----
        x_last = torch.zeros(B * self.N, self.total_input_dim, device=dev)
        tau_last = torch.zeros(B * self.N, self.total_input_dim, device=dev)

        outputs = []
        for t in range(self.lookback):
            # time index start at 1
            current_time = torch.full((B * self.N, 1), float(t + 1), device=dev)

            aux_t = X_aux_window[:, t].reshape(B * self.N, self.aux_dim)  # [B*N,11]
            ph_t = Y_ph_window[:, t].reshape(B * self.N, 1)               # [B*N,1]
            m_ph = mask_ph_window[:, t].reshape(B * self.N, 1)            # [B*N,1]

            z_aux = self.moe(aux_t)                                       # [B*N,moe_dim]
            x_t = torch.cat([ph_t, z_aux], dim=-1)                        # [B*N,D]

            # mask: aux always observed
            m_t = torch.cat([m_ph, torch.ones_like(z_aux)], dim=-1)       # [B*N,D]
            if self.input_mode == "x_only":
                m_t = torch.ones_like(m_t)
            elif self.input_mode == "mask_delta_only":
                x_t = torch.zeros_like(x_t)

            # Δt uses tau_{t-1}
            delta_t = current_time.expand(-1, self.total_input_dim) - tau_last + 1e-6
            if self.delta_clip_max is not None and self.delta_clip_max > 0:
                delta_t = torch.clamp(delta_t, min=0.0, max=self.delta_clip_max)
            else:
                delta_t = torch.clamp(delta_t, min=0.0)
            if self.delta_use_log1p:
                delta_t = torch.log1p(delta_t)

            # update tau_t / x_last with observed dims
            tau_t = torch.where(m_t > 0, current_time.expand(-1, self.total_input_dim), tau_last)
            x_last = torch.where(m_t > 0, x_t, x_last)

            # γ
            decay_rate = F.softplus(self.W_decay) * delta_t + F.softplus(self.b_decay)
            gamma = torch.exp(-decay_rate)  # [B*N,D]

            # x_tilde = γ⊙x_t + (1-γ)⊙x_last
            x_tilde = gamma * x_t + (1.0 - gamma) * x_last

            h = self.gru_cell(x_tilde, h)
            outputs.append(self.predictor(h))  # [B*N,H]

            tau_last = tau_t.detach()

        final_pred = outputs[-1].view(B, self.N, self.output_dim)  # [B,N,H]
        return final_pred


# ========================
# Build A_p, A_a from TRAIN only (DTW)
# ========================
def build_cluster_adjacency_from_train(Y_train_raw, X_train_raw, k_ph=4, k_aux=4, seed=0):
    """
    Y_train_raw: [TRAIN_DAYS, N]
    X_train_raw: [TRAIN_DAYS, N, 11]
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
    except Exception as e:
        raise ImportError(
            "需�?tslearn �?DTW 聚类构建 A_p/A_a。请安装：pip install tslearn"
        ) from e

    rng = np.random.RandomState(seed)
    N = Y_train_raw.shape[1]

    ph_series = Y_train_raw.astype(np.float32).T[:, :, None]  # [N,TRAIN,1]
    aux_series = np.nan_to_num(X_train_raw.astype(np.float32), nan=0.0).transpose(1, 0, 2)  # [N,TRAIN,11]

    km_ph = TimeSeriesKMeans(
        n_clusters=k_ph, metric="dtw", max_iter=DTW_MAX_ITER,
        n_init=2, random_state=rng.randint(0, 10**9)
    )
    labels_ph = km_ph.fit_predict(ph_series)

    km_aux = TimeSeriesKMeans(
        n_clusters=k_aux, metric="dtw", max_iter=DTW_MAX_ITER,
        n_init=2, random_state=rng.randint(0, 10**9)
    )
    labels_aux = km_aux.fit_predict(aux_series)

    A_p = (labels_ph[:, None] == labels_ph[None, :]).astype(np.float32)
    A_a = (labels_aux[:, None] == labels_aux[None, :]).astype(np.float32)
    return A_p, A_a, labels_ph, labels_aux


def _apply_missing_to_ph_window(ph_window, miss_rate, rng, missing_fill="zero"):
    """
    Apply synthetic missingness on a [W, N] pH window.
    missing_fill:
      - zero: masked value set to 0 (standardized mean)
      - ffill: masked value forward-filled within window; first missing stays 0
    """
    mask_window = np.ones_like(ph_window, dtype=np.float32)
    if miss_rate <= 0:
        return ph_window, mask_window

    missing = (rng.rand(*ph_window.shape) < miss_rate)
    ph_masked = ph_window.copy()
    mask_window[missing] = 0.0

    if missing_fill == "zero":
        ph_masked[missing] = 0.0
    elif missing_fill == "ffill":
        ph_masked[missing] = np.nan
        W, N = ph_masked.shape
        for n in range(N):
            last = 0.0
            for t in range(W):
                if np.isnan(ph_masked[t, n]):
                    ph_masked[t, n] = last
                else:
                    last = float(ph_masked[t, n])
    else:
        raise ValueError(f"Unsupported missing_fill: {missing_fill}")

    return ph_masked, mask_window


def rolling_forecast_std(
    model, X_all_std, Y_all_std, start_idx, end_idx, window, horizon,
    miss_rate, N, seed, device, missing_fill="zero"
):
    """
    Iterative rolling forecast on [start_idx, end_idx), returning predictions/targets in standardized scale.
    """
    if start_idx < window:
        raise ValueError(f"start_idx={start_idx} must be >= window={window}")
    if end_idx <= start_idx:
        raise ValueError(f"Invalid range: [{start_idx}, {end_idx})")

    y_roll_std = Y_all_std.copy()
    pred_blocks_std = []
    true_blocks_std = []
    step_blocks = []
    rng = np.random.RandomState(seed)

    with torch.no_grad():
        for t_start in range(start_idx, end_idx, horizon):
            block_len = min(horizon, end_idx - t_start)
            aux_window = X_all_std[t_start - window:t_start]   # [W,N,11]
            ph_window = y_roll_std[t_start - window:t_start]   # [W,N]
            ph_window, mask_window = _apply_missing_to_ph_window(
                ph_window, miss_rate=miss_rate, rng=rng, missing_fill=missing_fill
            )

            xb_aux = torch.from_numpy(aux_window[None, ...]).float().to(device)
            xb_ph = torch.from_numpy(ph_window[None, ...]).float().to(device)
            xb_mask = torch.from_numpy(mask_window[None, ...]).float().to(device)
            pred_block_std = model(xb_aux, xb_ph, xb_mask).cpu().numpy()[0]  # [N,H]

            y_roll_std[t_start:t_start + block_len] = pred_block_std[:, :block_len].T
            pred_blocks_std.append(pred_block_std[:, :block_len].T)      # [block_len,N]
            true_blocks_std.append(Y_all_std[t_start:t_start + block_len])  # [block_len,N]
            step_blocks.append(np.arange(1, block_len + 1, dtype=int))

    pred_all_std = np.concatenate(pred_blocks_std, axis=0)
    true_all_std = np.concatenate(true_blocks_std, axis=0)
    return pred_all_std, true_all_std, step_blocks


def apply_observation_dropout_batch(xb_ph, xb_mask, drop_rate):
    """
    Randomly hide a subset of currently observed pH inputs during training.
    """
    if drop_rate <= 0:
        return xb_ph, xb_mask
    drop = (torch.rand_like(xb_mask) < drop_rate) & (xb_mask > 0.5)
    xb_ph = xb_ph.masked_fill(drop, 0.0)
    xb_mask = xb_mask.masked_fill(drop, 0.0)
    return xb_ph, xb_mask


# ========================
# Main
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GRU-D (+MoE+3GAT) with optional ablations.")
    parser.add_argument("--seed-base", type=int, default=SEED_BASE,
                        help="Base seed used for repeat runs.")
    parser.add_argument("--input-modes", type=str, default=",".join(INPUT_MODES),
                        help="Comma-separated: full,x_only,mask_delta_only")
    parser.add_argument("--obs-dropout-rate", type=float, default=OBS_DROPOUT_RATE,
                        help="Extra train-time masking over observed pH values.")
    parser.add_argument("--delta-clip-max", type=float, default=DELTA_CLIP_MAX,
                        help="Clip upper bound for delta before optional log1p.")
    parser.add_argument("--delta-no-log1p", action="store_true",
                        help="Disable log1p transform on delta.")
    parser.add_argument("--train-miss-rate", type=float, default=0.0,
                        help="Fixed missing rate for training windows. Use 0.0 to train on clean history.")
    parser.add_argument("--missing-fill", type=str, default="zero", choices=["zero", "ffill"],
                        help="How to fill masked pH inputs before feeding model.")
    args = parser.parse_args()
    seed_base = args.seed_base

    input_modes = [m.strip() for m in args.input_modes.split(",") if m.strip()]
    valid_modes = {"full", "x_only", "mask_delta_only"}
    for m in input_modes:
        if m not in valid_modes:
            raise ValueError(f"Unsupported input mode: {m}. valid={sorted(valid_modes)}")

    # ---- Load data ----
    data = sio.loadmat(MAT_PATH)
    X_tr_cell = data["X_tr"]  # [1,423]
    Y_tr = data["Y_tr"]       # [37,423]
    X_te_cell = data["X_te"]  # [1,282]
    Y_te = data["Y_te"]       # [37,282]

    def stack_X(X_cell):
        days = X_cell.shape[1]
        return np.stack([X_cell[0, t] for t in range(days)], axis=0)  # [T,N,11]

    X_tr_all = stack_X(X_tr_cell)                       # [423,37,11]
    X_te_all = stack_X(X_te_cell)                       # [282,37,11]
    X_all = np.concatenate([X_tr_all, X_te_all], axis=0) # [705,37,11]

    Y_all = np.concatenate([Y_tr.T, Y_te.T], axis=0)     # [705,37]

    T_total, N, d_aux = X_all.shape
    print(f"Total T={T_total}, Sites N={N}, aux_dim={d_aux}")

    # ---- NO-LEAK: train raw ----
    X_train_raw = X_all[:TRAIN_DAYS]  # [423,N,11]
    Y_train_raw = Y_all[:TRAIN_DAYS]  # [423,N]

    # ---- NO-LEAK: static features ----
    F_p_raw_np = np.zeros((N, 3), dtype=np.float32)  # mean/std/slope on TRAIN only
    t_vals = np.arange(TRAIN_DAYS).reshape(-1, 1)
    for i in range(N):
        ph_series = Y_train_raw[:, i].astype(np.float32)
        mu = float(np.mean(ph_series))
        sigma = float(np.std(ph_series))
        lr_model = LinearRegression()
        lr_model.fit(t_vals, ph_series)
        slope = float(lr_model.coef_[0])
        F_p_raw_np[i] = [mu, sigma, slope]

    F_a_raw_np = np.mean(X_train_raw, axis=0).astype(np.float32)  # [N,11]
    F_a_raw_np = np.nan_to_num(F_a_raw_np, nan=0.0)

    # ---- NO-LEAK: build A_p, A_a from TRAIN only ----
    A_s = np.load(A_S_PATH).astype(np.float32)  # [N,N]
    if A_s.shape != (N, N):
        raise ValueError(f"A_s shape {A_s.shape} != (N,N)=({N},{N})")

    A_p, A_a, labels_ph, labels_aux = build_cluster_adjacency_from_train(
        Y_train_raw=Y_train_raw, X_train_raw=X_train_raw,
        k_ph=K_PH, k_aux=K_AUX, seed=seed_base
    )

    # edge_index to device (graph is used inside model on GPU)
    edge_index_s, _ = dense_to_sparse(torch.tensor(A_s))
    edge_index_p, _ = dense_to_sparse(torch.tensor(A_p))
    edge_index_a, _ = dense_to_sparse(torch.tensor(A_a))
    edge_index_s = edge_index_s.to(device)
    edge_index_p = edge_index_p.to(device)
    edge_index_a = edge_index_a.to(device)

    # ---- Experiments ----
    all_results = []
    prediction_rows = []

    for input_mode in input_modes:
        for horizon in HORIZONS:
            for miss_rate in MISSING_RATES:
                print(f"\n{'='*80}")
                print(
                    f"GRU-D+MoE+3GAT | mode={input_mode} | task=forecasting | "
                    f"horizon={horizon} | miss_rate={miss_rate} | repeats={REPEATS} | "
                    f"train_miss={args.train_miss_rate} | "
                    f"fill={args.missing_fill}"
                )
                print(f"{'='*80}")

                repeat_maes, repeat_rmses, repeat_mapes = [], [], []

                for rep in range(REPEATS):
                    seed = seed_base + rep
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed)

                    # ---- Standardize (NO-LEAK: fit only on TRAIN) ----
                    scalers_X = [StandardScaler() for _ in range(N)]
                    scalers_Y = [StandardScaler() for _ in range(N)]

                    X_all_std = np.zeros_like(X_all, dtype=np.float32)
                    Y_all_std = np.zeros_like(Y_all, dtype=np.float32)

                    for i in range(N):
                        X_all_std[:TRAIN_DAYS, i, :] = scalers_X[i].fit_transform(X_all[:TRAIN_DAYS, i, :])
                        X_all_std[TRAIN_DAYS:, i, :] = scalers_X[i].transform(X_all[TRAIN_DAYS:, i, :])

                        Y_all_std[:TRAIN_DAYS, i] = scalers_Y[i].fit_transform(Y_all[:TRAIN_DAYS, i:i+1]).ravel()
                        Y_all_std[TRAIN_DAYS:, i] = scalers_Y[i].transform(Y_all[TRAIN_DAYS:, i:i+1]).ravel()

                    # ---- Build sequences ----
                    T_valid = T_total - WINDOW - horizon + 1
                    if T_valid <= 0:
                        print(f"⚠️ 跳过：T_valid={T_valid} <= 0")
                        continue

                    X_seq_list, Y_hist_list, M_hist_list, Y_tgt_list = [], [], [], []

                    for i in range(T_valid):
                        aux_seq = X_all_std[i:i+WINDOW]     # [W,N,11]
                        ph_seq = Y_all_std[i:i+WINDOW]      # [W,N]

                        mask_seq = np.ones_like(ph_seq, dtype=np.float32)

                        train_miss_rate = float(args.train_miss_rate)
                        ph_seq, mask_seq = _apply_missing_to_ph_window(
                            ph_seq, miss_rate=train_miss_rate, rng=np.random, missing_fill=args.missing_fill
                        )

                        y_tgt = Y_all_std[i + WINDOW:i + WINDOW + horizon].T  # [N,H]

                        X_seq_list.append(aux_seq)
                        Y_hist_list.append(ph_seq)
                        M_hist_list.append(mask_seq)
                        Y_tgt_list.append(y_tgt)

                    X_seq = np.asarray(X_seq_list, dtype=np.float32)         # [T_valid,W,N,11]
                    Y_hist = np.asarray(Y_hist_list, dtype=np.float32)       # [T_valid,W,N]
                    M_hist = np.asarray(M_hist_list, dtype=np.float32)       # [T_valid,W,N]
                    Y_tgt = np.asarray(Y_tgt_list, dtype=np.float32)         # [T_valid,N,H]

                    # ---- Time split (NO-LEAK): target must lie in TRAIN for train/val ----
                    train_end = TRAIN_DAYS - WINDOW - horizon + 1  # last i whose target in TRAIN
                    if train_end <= VAL_SIZE + 1:
                        raise ValueError(f"Not enough training data: train_end={train_end}, VAL_SIZE={VAL_SIZE}")

                    T_train = train_end - VAL_SIZE
                    T_val = train_end
                    T_test = T_valid
                    val_roll_start = T_train + WINDOW
                    val_roll_end = TRAIN_DAYS
                    if val_roll_start >= val_roll_end:
                        raise ValueError(
                            f"Invalid val rolling range: start={val_roll_start}, end={val_roll_end}"
                        )

                    # �?IMPORTANT: keep these tensors on CPU (pin_memory expects CPU dense tensors)
                    X_train = torch.from_numpy(X_seq[:T_train])
                    Y_train_hist = torch.from_numpy(Y_hist[:T_train])
                    M_train_hist = torch.from_numpy(M_hist[:T_train])
                    Y_train_tgt = torch.from_numpy(Y_tgt[:T_train])

                    train_loader = DataLoader(
                        TensorDataset(X_train, Y_train_hist, M_train_hist, Y_train_tgt),
                        batch_size=BATCH_SIZE, shuffle=True,
                        pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS
                    )

                    # ---- Model ----
                    model = GRUDModel(
                        N=N, d_feat=D_FEAT, aux_dim=d_aux, moe_dim=MOE_DIM,
                        lookback=WINDOW, hidden_size=HIDDEN_SIZE,
                        num_experts=NUM_EXPERTS, num_heads=NUM_HEADS, output_dim=horizon,
                        input_mode=input_mode,
                        delta_clip_max=args.delta_clip_max,
                        delta_use_log1p=(not args.delta_no_log1p),
                    ).to(device)

                    model.set_static_info(
                        F_p_raw=torch.from_numpy(F_p_raw_np).float().to(device),
                        F_a_raw=torch.from_numpy(F_a_raw_np).float().to(device),
                        edge_index_s=edge_index_s,
                        edge_index_p=edge_index_p,
                        edge_index_a=edge_index_a,
                    )

                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    scaler = GradScaler(enabled=(device.type == "cuda"))

                    best_val = float("inf")
                    patience_cnt = 0
                    best_path = f"tmp_best_grud_{input_mode}_rep{rep}_h{horizon}_m{miss_rate}.pth"

                    # ---- Train ----
                    for epoch in range(EPOCHS):
                        model.train()
                        for xb_aux, xb_ph, xb_mask, yb in train_loader:
                            # �?move to GPU here
                            xb_aux = xb_aux.to(device, non_blocking=True)
                            xb_ph = xb_ph.to(device, non_blocking=True)
                            xb_mask = xb_mask.to(device, non_blocking=True)
                            yb = yb.to(device, non_blocking=True)

                            xb_ph, xb_mask = apply_observation_dropout_batch(
                                xb_ph, xb_mask, args.obs_dropout_rate
                            )

                            optimizer.zero_grad(set_to_none=True)
                            with autocast(enabled=(device.type == "cuda")):
                                pred = model(xb_aux, xb_ph, xb_mask)
                                loss = criterion(pred, yb)
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()

                        do_val = ((epoch + 1) % VAL_EVERY == 0) or (epoch == EPOCHS - 1)
                        if not do_val:
                            continue

                        # ---- Val (rolling forecast, every VAL_EVERY epochs) ----
                        model.eval()
                        val_pred_std, val_true_std, _ = rolling_forecast_std(
                            model=model,
                            X_all_std=X_all_std,
                            Y_all_std=Y_all_std,
                            start_idx=val_roll_start,
                            end_idx=val_roll_end,
                            window=WINDOW,
                            horizon=horizon,
                            miss_rate=float(args.train_miss_rate),
                            N=N,
                            seed=seed + 2000,
                            device=device,
                            missing_fill=args.missing_fill,
                        )
                        val_loss = float(np.mean((val_pred_std - val_true_std) ** 2))

                        if val_loss < best_val:
                            best_val = val_loss
                            patience_cnt = 0
                            torch.save(model.state_dict(), best_path)
                        else:
                            patience_cnt += 1
                            if patience_cnt >= PATIENCE:
                                break

                    # ---- Test ----
                    model.load_state_dict(torch.load(best_path, map_location=device))
                    if not KEEP_CHECKPOINTS:
                        try:
                            os.remove(best_path)
                        except OSError:
                            pass

                    model.eval()
                    pred_all_std, true_all_std, step_blocks = rolling_forecast_std(
                        model=model,
                        X_all_std=X_all_std,
                        Y_all_std=Y_all_std,
                        start_idx=TRAIN_DAYS,
                        end_idx=T_total,
                        window=WINDOW,
                        horizon=horizon,
                        miss_rate=miss_rate,
                        N=N,
                        seed=seed + 10000,
                        device=device,
                        missing_fill=args.missing_fill,
                    )

                    rmse_list, mae_list, mape_list = [], [], []
                    for i in range(N):
                        pred_i = scalers_Y[i].inverse_transform(pred_all_std[:, i:i+1]).reshape(-1)
                        true_i = scalers_Y[i].inverse_transform(true_all_std[:, i:i+1]).reshape(-1)

                        rmse = float(np.sqrt(mean_squared_error(true_i, pred_i)))
                        mae = float(mean_absolute_error(true_i, pred_i))
                        mape = float(mean_absolute_percentage_error(true_i, pred_i))

                        rmse_list.append(rmse)
                        mae_list.append(mae)
                        mape_list.append(mape)

                    offset = 0
                    for blk_steps in step_blocks:
                        block_len = len(blk_steps)
                        pred_blk = pred_all_std[offset:offset+block_len]
                        true_blk = true_all_std[offset:offset+block_len]
                        for row_i, st in enumerate(blk_steps):
                            for i in range(N):
                                yp = scalers_Y[i].inverse_transform(pred_blk[row_i, i:i+1].reshape(1, 1))[0, 0]
                                yt = scalers_Y[i].inverse_transform(true_blk[row_i, i:i+1].reshape(1, 1))[0, 0]
                                prediction_rows.append({
                                    "model": "GRUD",
                                    "input_mode": input_mode,
                                    "missing_rate": miss_rate,
                                    "horizon": horizon,
                                    "repeat": rep,
                                    "site": i,
                                    "step": int(st),
                                    "time_idx": int(offset + row_i),
                                    "y_true": float(yt),
                                    "y_pred": float(yp),
                                })
                        offset += block_len

                    avg_mae = float(np.mean(mae_list))
                    avg_rmse = float(np.mean(rmse_list))
                    avg_mape = float(np.mean(mape_list))

                    repeat_maes.append(avg_mae)
                    repeat_rmses.append(avg_rmse)
                    repeat_mapes.append(avg_mape)

                    print(f"[rep {rep+1}/{REPEATS}] MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.2f}%")

                final_mae = float(np.mean(repeat_maes))
                final_rmse = float(np.mean(repeat_rmses))
                final_mape = float(np.mean(repeat_mapes))

                all_results.append({
                    "input_mode": input_mode,
                    "missing_rate": miss_rate,
                    "horizon": horizon,
                    "task": "forecasting",
                    "mae": final_mae,
                    "rmse": final_rmse,
                    "mape": final_mape,
                })

                print(f"�?AVG over {REPEATS} reps: MAE={final_mae:.4f}, RMSE={final_rmse:.4f}, MAPE={final_mape:.2f}%")

    df = pd.DataFrame(all_results)[["input_mode", "missing_rate", "horizon", "task", "mae", "rmse", "mape"]]
    out_csv = "results_grud_moe_3gat_forecasting_noleak.csv"
    df.to_csv(out_csv, index=False)
    pd.DataFrame(prediction_rows).to_csv("predictions_GRUD_plot_ready.csv", index=False)
    print(f"\n🎉 完成！结果保存至 {out_csv}")


