import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 100

importance_tabpfn = pd.read_csv("./importance_tabpfn.csv")

new_names = ['Age at accelerometry, years','Long-standing illness, disability or infirmity','Body fat percentage','Number of correct matches in round',
             'Other eye problems','Number of treatments/medications taken','Current employment status',
             'Illness, injury, bereavement, stress in last 2 years (financial difficulties)','Time to complete round','Frequency of solarium/sunlamp use',
             'Mood swings','Illnesses of mother (Alzheimer\'s disease/dementia)','Diabetes diagnosed by doctor',
             'Alcohol usually taken with meals','Sex','Falls in the last year','Medication for pain relief, constipation, heartburn (laxatives)',
             'Mother still alive','Blood clot, DVT, bronchitis, emphysema\nasthma, rhinitis, eczema, allergy\ndiagnosed by doctor (blood clot in the leg (DVT))',
             'Body mass index']
len(new_names)
importance_tabpfn['new_names'] = new_names

fig, ax = plt.subplots(figsize=(12, 14))
top20_df_sorted = importance_tabpfn.iloc[::-1]

colors = plt.cm.viridis(top20_df_sorted['Importance_Mean'] / top20_df_sorted['Importance_Mean'].max())
bars = ax.barh(
    y=top20_df_sorted['new_names'],
    width=top20_df_sorted['Importance_Mean'],
    color=colors,
    edgecolor='black',
    linewidth=0.5
)
for bar in bars:
    width = bar.get_width()
    ax.text(
        x=width + width*0.01,
        y=bar.get_y() + bar.get_height()/2,
        s=f'{width:.6f}',
        va='center',
        ha='left',
        fontsize=9
    )

ax.set_xlabel('Importance of variables', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Primary care–feasible variables', fontsize=14, fontweight='bold', labelpad=10)
ax.grid(axis='x', linestyle='--', alpha=0.7, color='gray', linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis='x', labelsize=11)
ax.tick_params(axis='y', labelsize=10)
ax.set_xlim(right=top20_df_sorted['Importance_Mean'].max() * 1.15)
plt.tight_layout()
plt.savefig('../picture/xgb_top20_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import string
from sklearn.metrics import roc_curve, auc

models1 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_result/wearable-derived features + all primary care–feasible variables.npz"},
]

models2 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_exteral_result//results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_exteral_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_exteral_result/wearable-derived features + all primary care–feasible variables.npz"},
]

fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
panels = [
    {"ax": axes[0], "models": models1, "title": "Panel A: AUC curves for seven models developed\nusing data from England"},
    {"ax": axes[1], "models": models2, "title": "Panel B: AUC curves for external validation\nin Scotland and Wales"}
]
lines = []
for i, p in enumerate(panels):
    ax = p["ax"]
    for m in p["models"]:
        data = np.load(m["file"])
        y_true, y_score = data['y_true'], data['y_probs']
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        line, = ax.plot(fpr, tpr, lw=2, linestyle='--', label=m["name"])
        if i == 0: lines.append(line)

        # A, B, C, D, E, F, G
        label = string.ascii_uppercase[i] 

        ax.text(-0.1, 1.1, label, 
                transform=ax.transAxes,
                fontsize=16, 
                fontweight='bold', 
                va='top', 
                ha='right')

    ax.plot([0, 1], [0, 1], color='black', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('1 - Specificity', fontsize=10)
    ax.set_ylabel('Sensitivity', fontsize=10)
    # ax.set_title(p["title"], loc='left', fontweight='bold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout(rect=[0, 0.15, 1, 1])

fig.legend(handles=lines, 
           labels=[m["name"] for m in models1], 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.02), 
           ncol=3, 
           fontsize=9,
           frameon=False)

plt.savefig('../picture/AUC_test_ext.png', dpi=400, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from scipy.special import logit


def calculate_metrics_ci(y_true, y_probs, n_bootstraps=1000, seed=42):

    rng = np.random.RandomState(seed)
    bs_scores = []
    slope_scores = []
    intercept_scores = []


    y_probs_clipped = np.clip(y_probs, 1e-10, 1 - 1e-10)
    lp = logit(y_probs_clipped)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2:
            continue

        y_boot = y_true[indices]
        p_boot = y_probs[indices]
        lp_boot = lp[indices].reshape(-1, 1)

        # A. Brier Score
        bs_scores.append(brier_score_loss(y_boot, p_boot))

        lr = LogisticRegression(solver='lbfgs', C=1e10)
        lr.fit(lp_boot, y_boot)

        slope_scores.append(lr.coef_[0][0])
        intercept_scores.append(lr.intercept_[0])

    def get_ci(data):
        return np.mean(data), np.percentile(data, 2.5), np.percentile(data, 97.5)

    res = {
        "bs": get_ci(bs_scores),
        "slope": get_ci(slope_scores),
        "intercept": get_ci(intercept_scores)
    }
    return res

models1 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_result/wearable-derived features + all primary care–feasible variables.npz"},
]
models2 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_exteral_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_exteral_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_exteral_result/wearable-derived features + all primary care–feasible variables.npz"},
]

def perform_calibration(y_true, y_probs_raw, method='sigmoid'):
    y_probs_raw_reshed = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs')
    lr.fit(y_probs_raw_reshed, y_true)
    y_probs_calibrated = lr.predict_proba(y_probs_raw_reshed)[:, 1]
    return y_probs_calibrated

fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 8))

panels = [
    {"ax": axes[0], "models": models1, "title": "A"},
    {"ax": axes[1], "models": models2, "title": "B"}
]

legend_lines = []

for i, p in enumerate(panels):
    ax = p["ax"]
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Ideal", alpha=0.6) 

    for idx, m in enumerate(p["models"]):
        data = np.load(m["file"])
        y_true, y_probs_raw = data['y_true'], data['y_probs']

        y_probs_cal = perform_calibration(y_true, y_probs_raw, method='sigmoid')

        metrics = calculate_metrics_ci(y_true, y_probs_cal, n_bootstraps=1000)

        prob_true, prob_pred = calibration_curve(y_true, y_probs_cal, n_bins=10, strategy='quantile')

        line, = ax.plot(prob_pred, prob_true, marker='s', markersize=4,
                        lw=2, color=colors[idx], label=m["name"])
        if i == 0: legend_lines.append(line)

        print(f"[{p['title']}] {m['name']}:")
        print(f"  Brier Score: {metrics['bs'][0]:.8f} ({metrics['bs'][1]:.8f}-{metrics['bs'][2]:.8f})")
        print(f"  Slope:       {metrics['slope'][0]:.2f} ({metrics['slope'][1]:.2f}-{metrics['slope'][2]:.2f})")
        print(f"  Intercept:   {metrics['intercept'][0]:.2f} ({metrics['intercept'][1]:.2f}-{metrics['intercept'][2]:.2f})")
        print("-" * 30)

    display_limit = 0.05
    ax.set_xlim([-0.002, display_limit])
    ax.set_ylim([-0.002, display_limit])
    ax.set_xlabel('Predicted Probability (Calibrated)', fontsize=11)
    ax.set_ylabel('Observed Proportion', fontsize=11)
    ax.set_title(p["title"], fontweight='bold', fontsize=13, loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)

fig.legend(handles=legend_lines, labels=[m["name"] for m in models1], 
           loc='lower center', bbox_to_anchor=(0.5, 0.05), 
           ncol=2, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0.18, 1, 0.95])
plt.savefig('../picture/calibration_results_final.png', dpi=400, bbox_inches='tight')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import string
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from scipy.special import logit

def clean_and_flatten(input_data):
    if isinstance(input_data, (list, np.ndarray)) and len(input_data) > 0:
        try:
            flattened = np.concatenate(input_data).ravel()
        except:
            flattened = np.array([item for sublist in input_data for item in 
                                (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
    else:
        flattened = np.array(input_data).ravel()
    return flattened.astype(np.float64)

def calculate_all_metrics_ci(y_true, y_probs, n_bootstraps=1000, seed=42):

    rng = np.random.RandomState(seed)
    boot_results = {'auc': [], 'slope': [], 'intercept': [], 'brier': []}

    y_true = y_true.astype(int)

    y_probs_clipped = np.clip(y_probs, 1e-10, 1 - 1e-10)
    lp = logit(y_probs_clipped)

    for i in range(n_bootstraps):
        indices = rng.randint(0, len(y_true), len(y_true))
        if len(np.unique(y_true[indices])) < 2: continue

        y_boot = y_true[indices]
        p_boot = y_probs[indices]
        lp_boot = lp[indices].reshape(-1, 1)

        boot_results['auc'].append(roc_auc_score(y_boot, p_boot))

        boot_results['brier'].append(brier_score_loss(y_boot, p_boot))

        lr = LogisticRegression(solver='lbfgs', C=1e10)
        lr.fit(lp_boot, y_boot)
        boot_results['slope'].append(lr.coef_[0][0])
        boot_results['intercept'].append(lr.intercept_[0])

    output = {}
    for key, values in boot_results.items():
        mean_val = np.mean(values)
        low = np.percentile(values, 2.5)
        high = np.percentile(values, 97.5)
        fmt = ".8f" if key in ['brier'] else ".4f"
        output[key] = f"{mean_val:{fmt}} ({low:{fmt}}-{high:{fmt}})"

    return output

def perform_calibration(y_true, y_probs_raw):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2: return y_probs_raw
    y_probs_raw_res = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(y_probs_raw_res, y_true)
    return lr.predict_proba(y_probs_raw_res)[:, 1]

windows = ["0-2", "0-4", "0-6"]
data_dirs = [
    {"title": "Internal Testing", "path": "./AUC_results_internal_England"},
    {"title": "External Validation", "path": "./AUC_results_external_Validation"}
]
model_configs = [
    {"name": "Only accelerometer-derived features", "prefix": "Only_wearable"},
    {"name": "Only all primary care–feasible variables", "prefix": "Only_all_primary"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "prefix": "Top1_+_wearable"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "prefix": "Top5_+_wearable"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "prefix": "Top10_+_wearable"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "prefix": "Top15_+_wearable"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "prefix": "Top20_+_wearable"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "prefix": "All_+_wearable"}
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
legend_lines = [] 

for row, d_cfg in enumerate(data_dirs):
    for col, win in enumerate(windows):
        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)

        print(f"\n" + "="*50)
        print(f"Panel {string.ascii_uppercase[row*3+col]}: {d_cfg['title']} | Window: {win} Years")
        print("="*50)

        for i, m_cfg in enumerate(model_configs):
            file_path = f"{d_cfg['path']}/{m_cfg['prefix']}_{win}_years.npz"
            try:
                data = np.load(file_path, allow_pickle=True)
                yt = clean_and_flatten(data['y_true'])
                yp = clean_and_flatten(data['y_probs'])
                mask = ~np.isnan(yt) & ~np.isnan(yp)
                y_true, y_probs_raw = yt[mask].astype(int), yp[mask]

                if len(y_true) < 10: continue

                y_probs_cal = perform_calibration(y_true, y_probs_raw)

                metrics = calculate_all_metrics_ci(y_true, y_probs_cal)

                prob_true, prob_pred = calibration_curve(y_true, y_probs_cal, n_bins=8, strategy='quantile')

                line, = ax.plot(prob_pred, prob_true, color=colors[i], marker='s', markersize=3,
                                lw=1.5, label=m_cfg["name"])

                if row == 0 and col == 0:
                    legend_lines.append(line)

                print(f"Model: {m_cfg['name'][:30]}...")
                print(f"  AUC:       {metrics['auc']}")
                print(f"  Slope:     {metrics['slope']}")
                print(f"  Intercept: {metrics['intercept']}")
                print(f"  Brier:     {metrics['brier']}")
                print("-" * 20)

            except Exception as e:
                print(f"  [Error] {file_path}: {e}")

        if win == "0-2": display_limit = 0.02
        elif win == "0-4": display_limit = 0.015
        else: display_limit = 0.03

        ax.set_xlim([0, display_limit])
        ax.set_ylim([0, display_limit])
        ax.set_xlabel('Predicted Probability', fontsize=11)
        ax.set_ylabel('Observed Proportion', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)

        panel_idx = row * 3 + col
        ax.text(0, 1.05, string.ascii_uppercase[panel_idx], transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='bottom', ha='right')

        if col == 1:
            ax.set_title(d_cfg["title"], fontsize=22, fontweight='bold', pad=35)


plt.tight_layout(rect=[0.02, 0.15, 0.98, 0.95])

fig.legend(handles=legend_lines, 
           labels=[m["name"] for m in model_configs], 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.04), 
           ncol=2, 
           fontsize=10,
           frameon=False)

plt.savefig('../picture/calibration_complete_metrics.png', dpi=400, bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def perform_calibration(y_true, y_probs_raw, method='sigmoid'):
    y_probs_raw_reshed = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs')
    lr.fit(y_probs_raw_reshed, y_true)
    y_probs_calibrated = lr.predict_proba(y_probs_raw_reshed)[:, 1]
    return y_probs_calibrated

def calculate_net_benefit(y_true, y_probs, thresholds):
    net_benefits = []
    N = len(y_true)

    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
            continue
        preds = (y_probs >= pt).astype(int)

        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))

        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)

def calculate_treat_all(y_true, thresholds):
    N = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    net_benefits = []
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
        else:
            nb = (tp / N) - (fp / N) * (pt / (1 - pt))
            net_benefits.append(nb)
    return np.array(net_benefits)
models1 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_result/wearable-derived features + all primary care–feasible variables.npz"},
]
models2 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_exteral_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_exteral_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_exteral_result/wearable-derived features + all primary care–feasible variables.npz"},
]

fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 8))

panels = [
    {"ax": axes[0], "models": models1, "title": "A: Internal Testing"},
    {"ax": axes[1], "models": models2, "title": "B: External Validation"}
]

thresholds = np.linspace(0.001, 0.05, 100)

legend_lines = []

for i, p in enumerate(panels):
    ax = p["ax"]

    y_true_base = None

    for idx, m in enumerate(p["models"]):
        data = np.load(m["file"])
        y_true, y_probs_raw = data['y_true'], data['y_probs']

        if y_true_base is None:
            y_true_base = y_true

        y_probs_cal = perform_calibration(y_true, y_probs_raw, method='sigmoid')

        net_benefit = calculate_net_benefit(y_true, y_probs_cal, thresholds)

        line, = ax.plot(thresholds, net_benefit, lw=2, color=colors[idx], label=m["name"])
        if i == 0: legend_lines.append(line)

    if y_true_base is not None:
        nb_treat_all = calculate_treat_all(y_true_base, thresholds)

        line_all, = ax.plot(thresholds, nb_treat_all, "k--", lw=1.5, label="Treat All")
        line_none, = ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1.5, label="Treat None")

        if i == 0:
            legend_lines.extend([line_all, line_none])

    ax.set_xlim([0, 0.05])

    y_min = max(np.min(nb_treat_all) * 1.1, -0.01)
    y_max = np.max(nb_treat_all) * 1.2
    ax.set_ylim([y_min, y_max])

    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(p["title"], fontweight='bold', fontsize=14, loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)

fig.legend(handles=legend_lines, 
           labels=[m["name"] for m in models1] + ["Treat All", "Treat None"], 
           loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=3, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('../picture/DCA_results_final.png', dpi=400, bbox_inches='tight')
plt.show()

import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def clean_and_flatten(input_data):
    if isinstance(input_data, (list, np.ndarray)) and len(input_data) > 0:
        try:
            flattened = np.concatenate(input_data).ravel()
        except:
            flattened = np.array([item for sublist in input_data for item in 
                                (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
    else:
        flattened = np.array(input_data).ravel()
    return flattened.astype(np.float64)

def perform_calibration(y_true, y_probs_raw):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2: return y_probs_raw
    y_probs_raw_res = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(y_probs_raw_res, y_true)
    return lr.predict_proba(y_probs_raw_res)[:, 1]


def calculate_net_benefit(y_true, y_probs, thresholds):
    net_benefits = []
    N = len(y_true)
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
            continue
        preds = (y_probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def calculate_treat_all(y_true, thresholds):
    N = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    net_benefits = []
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
        else:
            nb = (tp / N) - (fp / N) * (pt / (1 - pt))
            net_benefits.append(nb)
    return np.array(net_benefits)

windows = ["0-2", "0-4", "0-6"]
data_dirs = [
    {"title": "Internal Testing", "path": "./AUC_results_internal_England"},
    {"title": "External Validation", "path": "./AUC_results_external_Validation"}
]
model_configs = [
    {"name": "Only accelerometer-derived features", "prefix": "Only_wearable"},
    {"name": "Only all primary care–feasible variables", "prefix": "Only_all_primary"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "prefix": "Top1_+_wearable"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "prefix": "Top5_+_wearable"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "prefix": "Top10_+_wearable"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "prefix": "Top15_+_wearable"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "prefix": "Top20_+_wearable"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "prefix": "All_+_wearable"}
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
legend_lines = [] 

for row, d_cfg in enumerate(data_dirs):
    for col, win in enumerate(windows):
        ax = axes[row, col]

        if win == "0-2": display_limit = 0.015
        elif win == "0-4": display_limit = 0.025
        else: display_limit = 0.04

        thresholds = np.linspace(0.0001, display_limit, 100)
        y_true_base = None

        print(f"\n" + "="*50)
        print(f"Panel {string.ascii_uppercase[row*3+col]}: {d_cfg['title']} | Window: {win} Years")
        print("="*50)

        for i, m_cfg in enumerate(model_configs):
            file_path = f"{d_cfg['path']}/{m_cfg['prefix']}_{win}_years.npz"
            try:
                data = np.load(file_path, allow_pickle=True)
                yt = clean_and_flatten(data['y_true'])
                yp = clean_and_flatten(data['y_probs'])
                mask = ~np.isnan(yt) & ~np.isnan(yp)
                y_true, y_probs_raw = yt[mask].astype(int), yp[mask]

                if len(y_true) < 10: continue
                if y_true_base is None: y_true_base = y_true

                y_probs_cal = perform_calibration(y_true, y_probs_raw)

                net_benefit = calculate_net_benefit(y_true, y_probs_cal, thresholds)

                line, = ax.plot(thresholds, net_benefit, color=colors[i], lw=2, label=m_cfg["name"])

                if row == 0 and col == 0:
                    legend_lines.append(line)

                print(f"Model: {m_cfg['name'][:30]}... DCA computed.")

            except Exception as e:
                print(f"  [Error] {file_path}: {e}")

        if y_true_base is not None:
            nb_treat_all = calculate_treat_all(y_true_base, thresholds)
            line_all, = ax.plot(thresholds, nb_treat_all, "k--", lw=1.5, label="Treat All")
            line_none, = ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1.5, label="Treat None")

            if row == 0 and col == 0:
                legend_lines.extend([line_all, line_none])

        ax.set_xlim([0, display_limit])

        if y_true_base is not None:
            max_nb = np.max(calculate_treat_all(y_true_base, [0.0001]))
            ax.set_ylim([-max_nb * 0.25, max_nb * 1.3])

        ax.set_xlabel('Threshold Probability', fontsize=11)
        ax.set_ylabel('Net Benefit', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)

        panel_idx = row * 3 + col
        ax.text(0, 1.05, string.ascii_uppercase[panel_idx], transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='bottom', ha='right')

        if row == 0:
            ax.set_title(f"Window: {win} Years", fontsize=16, fontweight='bold', pad=15)

        if col == 2:
            ax.text(1.05, 0.5, d_cfg["title"], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', rotation=270, va='center', ha='left')

plt.tight_layout(rect=[0.02, 0.15, 0.95, 0.95])

labels = [m["name"] for m in model_configs] + ["Treat All", "Treat None"]
fig.legend(handles=legend_lines, 
           labels=labels, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.02), 
           ncol=3, 
           fontsize=11,
           frameon=False)

plt.savefig('../picture/DCA_complete_6panels.png', dpi=400, bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def perform_calibration(y_true, y_probs_raw, method='sigmoid'):
    y_probs_raw_reshed = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs')
    lr.fit(y_probs_raw_reshed, y_true)
    y_probs_calibrated = lr.predict_proba(y_probs_raw_reshed)[:, 1]
    return y_probs_calibrated

def calculate_net_benefit(y_true, y_probs, thresholds):
    net_benefits = []
    N = len(y_true)
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
            continue
        preds = (y_probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def calculate_treat_all(y_true, thresholds):
    N = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    net_benefits = []
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
        else:
            nb = (tp / N) - (fp / N) * (pt / (1 - pt))
            net_benefits.append(nb)
    return np.array(net_benefits)

models1 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_result/wearable-derived features + all primary care–feasible variables.npz"},
]
models2 = [
    {"name": "Only accelerometer-derived features", "file": "./AUC_exteral_result/results_wearable_only.npz"},
    {"name": "Only all primary care–feasible variables", "file": "./AUC_exteral_result/Only all primary care–feasible variables.npz"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 1 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 5 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 10 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 15 selected variables.npz"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "file": "./AUC_exteral_result/Wearable-derived features + top 20 selected variables.npz"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "file": "./AUC_exteral_result/wearable-derived features + all primary care–feasible variables.npz"},
]

fig, axes = plt.subplots(1, 2, figsize=(16, 8), dpi=300)
colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, 8))

panels = [
    {"ax": axes[0], "models": models1, "title": "A: Internal Testing"},
    {"ax": axes[1], "models": models2, "title": "B: External Validation"}
]

thresholds = np.linspace(0.001, 0.05, 100) 
legend_lines = []

for i, p in enumerate(panels):
    ax = p["ax"]
    y_true_base = None

    for idx, m in enumerate(p["models"]):
        data = np.load(m["file"])
        y_true, y_probs_raw = data['y_true'], data['y_probs']

        if y_true_base is None:
            y_true_base = y_true

        y_probs_cal = perform_calibration(y_true, y_probs_raw)
        net_benefit = calculate_net_benefit(y_true, y_probs_cal, thresholds)

        avg_net_benefit = np.mean(net_benefit)
        max_net_benefit = np.max(net_benefit)
        print(f"\n{'='*60}")
        print(f"Dataset：{p['title']}")
        print(f"Model：{m['name']}")
        print(f"Average net income (all thresholds) = {avg_net_benefit:.6f}")
        print(f"Maximum net profit (optimal threshold) = {max_net_benefit:.6f}")
        print(f"{'='*60}")
        # ==================================================================

        line, = ax.plot(thresholds, net_benefit, lw=2, color=colors[idx], label=m["name"])
        if i == 0: legend_lines.append(line)

    if y_true_base is not None:
        nb_treat_all = calculate_treat_all(y_true_base, thresholds)
        line_all, = ax.plot(thresholds, nb_treat_all, "k--", lw=1.5, label="Treat All")
        line_none, = ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1.5, label="Treat None")
        if i == 0:
            legend_lines.extend([line_all, line_none])

    ax.set_xlim([0, 0.05])
    y_min = max(np.min(nb_treat_all) * 1.1, -0.01) 
    y_max = np.max(nb_treat_all) * 1.2
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('Threshold Probability', fontsize=12)
    ax.set_ylabel('Net Benefit', fontsize=12)
    ax.set_title(p["title"], fontweight='bold', fontsize=14, loc='left')
    ax.grid(True, linestyle=':', alpha=0.5)

fig.legend(handles=legend_lines, 
           labels=[m["name"] for m in models1] + ["Treat All", "Treat None"], 
           loc='lower center', bbox_to_anchor=(0.5, -0.05), 
           ncol=3, fontsize=10, frameon=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
# plt.savefig('../picture/DCA_results_final.png', dpi=400, bbox_inches='tight')
plt.show()


import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

def clean_and_flatten(input_data):
    if isinstance(input_data, (list, np.ndarray)) and len(input_data) > 0:
        try:
            flattened = np.concatenate(input_data).ravel()
        except:
            flattened = np.array([item for sublist in input_data for item in 
                                (sublist if isinstance(sublist, (list, np.ndarray)) else [sublist])])
    else:
        flattened = np.array(input_data).ravel()
    return flattened.astype(np.float64)

def perform_calibration(y_true, y_probs_raw):
    y_true = y_true.astype(int)
    if len(np.unique(y_true)) < 2: return y_probs_raw
    y_probs_raw_res = y_probs_raw.reshape(-1, 1)
    lr = LogisticRegression(C=1e10, solver='lbfgs', max_iter=1000)
    lr.fit(y_probs_raw_res, y_true)
    return lr.predict_proba(y_probs_raw_res)[:, 1]

def calculate_net_benefit(y_true, y_probs, thresholds):
    net_benefits = []
    N = len(y_true)
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
            continue
        preds = (y_probs >= pt).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(nb)
    return np.array(net_benefits)

def calculate_treat_all(y_true, thresholds):
    N = len(y_true)
    tp = np.sum(y_true == 1)
    fp = np.sum(y_true == 0)
    net_benefits = []
    for pt in thresholds:
        if pt >= 1.0:
            net_benefits.append(0)
        else:
            nb = (tp / N) - (fp / N) * (pt / (1 - pt))
            net_benefits.append(nb)
    return np.array(net_benefits)

windows = ["0-2", "0-4", "0-6"]
data_dirs = [
    {"title": "Internal Testing", "path": "./AUC_results_internal_England"},
    {"title": "External Validation", "path": "./AUC_results_external_Validation"}
]
model_configs = [
    {"name": "Only accelerometer-derived features", "prefix": "Only_wearable"},
    {"name": "Only all primary care–feasible variables", "prefix": "Only_all_primary"},
    {"name": "Accelerometer-derived features + top 1 selected variables", "prefix": "Top1_+_wearable"},
    {"name": "Accelerometer-derived features + top 5 selected variables", "prefix": "Top5_+_wearable"},
    {"name": "Accelerometer-derived features + top 10 selected variables", "prefix": "Top10_+_wearable"},
    {"name": "Accelerometer-derived features + top 15 selected variables", "prefix": "Top15_+_wearable"},
    {"name": "Accelerometer-derived features + top 20 selected variables", "prefix": "Top20_+_wearable"},
    {"name": "Accelerometer-derived features + all primary care–feasible variables", "prefix": "All_+_wearable"}
]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

fig, axes = plt.subplots(2, 3, figsize=(20, 12), dpi=150)
legend_lines = [] 

for row, d_cfg in enumerate(data_dirs):
    for col, win in enumerate(windows):
        ax = axes[row, col]

        if win == "0-2": display_limit = 0.015
        elif win == "0-4": display_limit = 0.025
        else: display_limit = 0.04

        thresholds = np.linspace(0.0001, display_limit, 100)
        y_true_base = None 

        print(f"\n" + "="*70)
        print(f"Panel {string.ascii_uppercase[row*3+col]}: {d_cfg['title']} | Window: {win} Years")
        print("="*70)

        for i, m_cfg in enumerate(model_configs):
            file_path = f"{d_cfg['path']}/{m_cfg['prefix']}_{win}_years.npz"
            try:
                data = np.load(file_path, allow_pickle=True)
                yt = clean_and_flatten(data['y_true'])
                yp = clean_and_flatten(data['y_probs'])
                mask = ~np.isnan(yt) & ~np.isnan(yp)
                y_true, y_probs_raw = yt[mask].astype(int), yp[mask]

                if len(y_true) < 10: continue
                if y_true_base is None: y_true_base = y_true

                y_probs_cal = perform_calibration(y_true, y_probs_raw)

                net_benefit = calculate_net_benefit(y_true, y_probs_cal, thresholds)

                avg_net_benefit = np.mean(net_benefit)
                max_net_benefit = np.max(net_benefit)

                print(f"Model: {m_cfg['name']}")
                print(f"  ├─ Average net income (Avg) = {avg_net_benefit:.6f}")
                print(f"  └─ Maximum net income (Max) = {max_net_benefit:.6f}")
                print("-" * 70)

                line, = ax.plot(thresholds, net_benefit, color=colors[i], lw=2, label=m_cfg["name"])

                if row == 0 and col == 0:
                    legend_lines.append(line)

            except Exception as e:
                print(f"  [Error] {file_path}: {e}")
                print("-" * 70)

        if y_true_base is not None:
            nb_treat_all = calculate_treat_all(y_true_base, thresholds)
            line_all, = ax.plot(thresholds, nb_treat_all, "k--", lw=1.5, label="Treat All")
            line_none, = ax.plot(thresholds, np.zeros_like(thresholds), "k:", lw=1.5, label="Treat None")

            if row == 0 and col == 0:
                legend_lines.extend([line_all, line_none])

        ax.set_xlim([0, display_limit])

        if y_true_base is not None:
            max_nb = np.max(calculate_treat_all(y_true_base, [0.0001]))
            ax.set_ylim([-max_nb * 0.25, max_nb * 1.3])

        ax.set_xlabel('Threshold Probability', fontsize=11)
        ax.set_ylabel('Net Benefit', fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.4)

        panel_idx = row * 3 + col
        ax.text(0, 1.05, string.ascii_uppercase[panel_idx], transform=ax.transAxes, 
                fontsize=18, fontweight='bold', va='bottom', ha='right')

        if row == 0:
            ax.set_title(f"Window: {win} Years", fontsize=16, fontweight='bold', pad=15)

        if col == 2:
            ax.text(1.05, 0.5, d_cfg["title"], transform=ax.transAxes,
                    fontsize=18, fontweight='bold', rotation=270, va='center', ha='left')

plt.tight_layout(rect=[0.02, 0.15, 0.95, 0.95])

labels = [m["name"] for m in model_configs] + ["Treat All", "Treat None"]
fig.legend(handles=legend_lines, 
           labels=labels, 
           loc='lower center', 
           bbox_to_anchor=(0.5, 0.02), 
           ncol=3, 
           fontsize=11,
           frameon=False)

# plt.savefig('../picture/DCA_complete_6panels.png', dpi=400, bbox_inches='tight')
plt.show()
