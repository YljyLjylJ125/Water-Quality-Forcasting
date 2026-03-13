import os
import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import cdist_dtw

# =========================================================
# ✅ 0️⃣ 参数（只用训练集 + 6缺失率 + DTW距离 + 层次聚类）
# =========================================================
K_RANGE = range(2, 10)  # 你之前用的是2~9
MISSING_RATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# 只用训练集；如果你还想去掉“验证尾段”（423里去掉最后82天），改 True
DROP_VAL_TAIL = False
VAL_SIZE = 82

# 层次聚类linkage：推荐 average 或 complete
LINKAGE = "average"

# 缺失注入随机种子（保证可复现）
BASE_SEED = 2026

# 画图样式
MEAN_LW = 2.5
BAND_ALPHA = 0.20

# 是否保存每个缺失率的聚类结果csv
SAVE_EACH_CSV = True

# 是否保存图片
SAVE_FIGS = True
FIG_DPI = 200


# =========================================================
# ✅ 1️⃣ 载入数据：只用训练集 Y_tr
# =========================================================
dataset_candidates = [
    "/root/multi-water-quality/datasets/water_dataset.mat",
    "/root/multi-water-quality/datasets/water_dataset.mat",
    "water_dataset.mat",
]
dataset_path = next((p for p in dataset_candidates if os.path.exists(p)), None)
if dataset_path is None:
    raise FileNotFoundError("Cannot find water_dataset.mat")
data = sio.loadmat(dataset_path)
Y_tr = data["Y_tr"]  # (37, 423)

if DROP_VAL_TAIL:
    Y_tr = Y_tr[:, : (Y_tr.shape[1] - VAL_SIZE)]  # (37, 341)

n_points, n_days = Y_tr.shape
print(f"✅ 使用训练集聚类：{n_points}个站点, {n_days}天（只用Y_tr）")


# =========================================================
# ✅ 2️⃣ 缺失注入：随机置缺失，然后用“站点均值”回填（DTW不能有NaN）
# =========================================================
def inject_missing_and_impute_site_mean(Y, miss_rate, seed):
    if miss_rate <= 0:
        return Y.copy().astype(float)

    rng = np.random.default_rng(seed)
    mask = rng.random(Y.shape) < miss_rate  # True=missing

    Y_imp = Y.copy().astype(float)
    global_fallback = float(np.mean(Y_imp))

    for i in range(Y.shape[0]):
        obs = Y_imp[i, ~mask[i]]
        fill_value = float(np.mean(obs)) if obs.size > 0 else global_fallback
        Y_imp[i, mask[i]] = fill_value

    # 可选：打印实际缺失比例
    # print(f"  applied missing ratio ~ {mask.mean():.4f}")
    return Y_imp


# =========================================================
# ✅ 3️⃣ 画图函数：某缺失率的“簇均值±std”
# =========================================================
def plot_cluster_mean_std_on_ax(ax, Y_used, labels, miss_rate, date_vals):
    uniq = sorted(np.unique(labels))
    n_days = Y_used.shape[1]

    for c in uniq:
        idx = np.where(labels == c)[0]
        curves = Y_used[idx]  # (n_members, n_days)
        mean_curve = np.mean(curves, axis=0)
        std_curve = np.std(curves, axis=0)

        # 阴影跟随线颜色（避免“某色线没阴影”）
        line, = ax.plot(date_vals, mean_curve, linewidth=MEAN_LW, label=f"C{c} (n={len(idx)})")
        ax.fill_between(
            date_vals,
            mean_curve - std_curve,
            mean_curve + std_curve,
            color=line.get_color(),
            alpha=BAND_ALPHA
        )

    ax.set_title(f"MR={miss_rate}", fontsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(alpha=0.2)


# =========================================================
# ✅ 4️⃣ 层次聚类：用预计算DTW距离聚类 + silhouette选k
# =========================================================
def agglo_fit_predict_precomputed(dist_matrix, k, linkage=LINKAGE):
    # sklearn版本兼容：metric vs affinity
    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            metric="precomputed",
            linkage=linkage
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=k,
            affinity="precomputed",
            linkage=linkage
        )
    return model.fit_predict(dist_matrix)


# =========================================================
# ✅ 5️⃣ 主流程：6个缺失率分别聚类 + 单独画6张 + 再合成1张2×3
# =========================================================
summary_rows = []
start_date = datetime(2016, 2, 28)
date_vals = np.array([start_date + timedelta(days=i) for i in range(n_days)])

# 先准备汇总2×3图
fig_sum, axes_sum = plt.subplots(2, 3, figsize=(18, 9), sharex=False, sharey=False)
axes_sum = np.array(axes_sum).reshape(-1)

for j, miss_rate in enumerate(MISSING_RATES):
    print("\n" + "=" * 70)
    print(f"🔁 miss_rate={miss_rate}: DTW距离 + 层次聚类（只用训练集）")
    print("=" * 70)

    seed = BASE_SEED + int(miss_rate * 1000)
    Y_miss = inject_missing_and_impute_site_mean(Y_tr, miss_rate, seed)

    # tslearn DTW 输入需要 (n_ts, sz, d)
    Y_3d = Y_miss.reshape(n_points, n_days, 1)

    # DTW距离矩阵
    dist_matrix = cdist_dtw(Y_3d)  # (37,37)

    # silhouette选k（保持你原逻辑：遍历K_RANGE选最优）
    best_k, best_score, best_labels = None, -1, None
    for k in K_RANGE:
        labels_k = agglo_fit_predict_precomputed(dist_matrix, k, linkage=LINKAGE)
        score_k = silhouette_score(dist_matrix, labels_k, metric="precomputed")
        print(f"  k={k:2d} → Silhouette = {score_k:.3f}")
        if score_k > best_score:
            best_score, best_k, best_labels = score_k, k, labels_k

    print(f"✅ miss={miss_rate} 最佳k={best_k} (silhouette={best_score:.3f})")

    # 输出簇成员
    cluster_df = pd.DataFrame({"Site": np.arange(1, n_points + 1), "Cluster": best_labels})
    for c in sorted(cluster_df["Cluster"].unique()):
        members = cluster_df[cluster_df["Cluster"] == c]["Site"].tolist()
        print(f"  Cluster {c}: {members}")

    # 保存每个缺失率的csv（可选）
    if SAVE_EACH_CSV:
        out_csv = f"ph_dtw_agglo_trainonly_miss{miss_rate:.1f}.csv"
        cluster_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"💾 已保存：{out_csv}")

    summary_rows.append({"missing_rate": miss_rate, "best_k": best_k, "silhouette": best_score})

    # ========== (A) 单独画一张 ==========
    fig_one, ax_one = plt.subplots(1, 1, figsize=(12, 5))
    plot_cluster_mean_std_on_ax(ax_one, Y_miss, best_labels, miss_rate, date_vals)
    ax_one.legend(loc="best", frameon=False)
    ax_one.set_xlabel("")
    ax_one.set_ylabel("pH")
    ax_one.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    if SAVE_FIGS:
        out_png = f"ph_cluster_miss{miss_rate:.1f}_single.png"
        fig_one.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
        print(f"🖼️ 已保存：{out_png}")

    plt.show()

    # ========== (B) 画到汇总2×3里 ==========
    ax_sum = axes_sum[j]
    plot_cluster_mean_std_on_ax(ax_sum, Y_miss, best_labels, miss_rate, date_vals)
    ax_sum.legend(loc="best", frameon=False, fontsize=9)
    ax_sum.tick_params(axis="x", rotation=30)
    ax_sum.set_xlabel("")
    if j % 3 == 0:
        ax_sum.set_ylabel("pH")
    else:
        ax_sum.set_ylabel("")

# 关闭多余子图（这里正好6格用满）
for k in range(len(MISSING_RATES), len(axes_sum)):
    axes_sum[k].axis("off")

# 汇总图只放一个legend（从第一个子图拿）
plt.tight_layout()
fig_sum.subplots_adjust(hspace=0.34)

if SAVE_FIGS:
    out_png = "ph_cluster_all_missing_2x3.png"
    fig_sum.savefig(out_png, dpi=FIG_DPI, bbox_inches="tight")
    print(f"\n🖼️ 已保存汇总图：{out_png}")

plt.show()

# 保存汇总表（每个缺失率的best_k/silhouette）
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("ph_cluster_summary_by_missing_rate.csv", index=False, encoding="utf-8-sig")
print("\n✅ 已保存：ph_cluster_summary_by_missing_rate.csv")
print(summary_df)
