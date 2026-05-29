import numpy as np
import pandas as pd
import os
import scipy.stats as stats
from itertools import combinations

def get_auc_and_variance(ground_truth, predictions):
    if isinstance(ground_truth, np.ndarray) and ground_truth.dtype == 'O':
        ground_truth = np.concatenate(ground_truth)
    if isinstance(predictions, np.ndarray) and predictions.dtype == 'O':
        predictions = np.concatenate(predictions)

    gt = np.asarray(ground_truth).flatten().astype(float)
    pr = np.asarray(predictions).flatten().astype(float)

    order = np.argsort(pr)
    gt_s, pr_s = gt[order], pr[order]

    pos_idx = np.where(gt_s == 1)[0]
    neg_idx = np.where(gt_s == 0)[0]
    m, n = len(pos_idx), len(neg_idx)

    if m == 0 or n == 0: return 0.5, 0.0

    v01 = np.zeros(m)
    v10 = np.zeros(n)
    for i in range(m):
        v01[i] = np.sum(pr_s[neg_idx] < pr_s[pos_idx[i]]) / n
    for j in range(n):
        v10[j] = np.sum(pr_s[pos_idx] > pr_s[neg_idx[j]]) / m

    auc = np.mean(v01)
    var = np.var(v01) / m + np.var(v10) / n
    return auc, var

def compare_windows(model, win1, win2, folder):
    path1 = os.path.join(folder, f"{model}_{win1}.npz")
    path2 = os.path.join(folder, f"{model}_{win2}.npz")

    if not (os.path.exists(path1) and os.path.exists(path2)):
        return None

    d1 = np.load(path1, allow_pickle=True)
    auc1, var1 = get_auc_and_variance(d1['y_true'], d1['y_probs'])

    d2 = np.load(path2, allow_pickle=True)
    auc2, var2 = get_auc_and_variance(d2['y_true'], d2['y_probs'])

    diff = auc1 - auc2
    se = np.sqrt(var1 + var2)
    z = diff / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))

    return {
        'Model': model,
        'Type': 'Internal' if 'internal' in folder.lower() else 'External',
        'Comparison': f"{win1} vs {win2}",
        'AUC_1': round(auc1, 4),
        'AUC_2': round(auc2, 4),
        'Diff': round(diff, 4),
        'P_Value': f"{p_value:.6f}",
        'Significant': 'Yes' if p_value < 0.05 else 'No'
    }


windows = ['0-2_years', '0-4_years', '0-6_years']
window_pairs = list(combinations(windows, 2))
folders = ['AUC_results_internal_England', 'AUC_results_external_Validation']
models = [
    'Only_wearable', 'Only_all_primary', 'Top1_+_wearable', 'Top5_+_wearable', 
    'Top10_+_wearable', 'Top15_+_wearable', 'Top20_+_wearable', 'All_+_wearable'
]

final_results = []

for folder in folders:
    for model in models:
        for w1, w2 in window_pairs:
            res = compare_windows(model, w1, w2, folder)
            if res:
                final_results.append(res)

df = pd.DataFrame(final_results)
df.to_csv('./Statistical analysis/DeLong_Window_Comparison.csv', index=False, encoding='utf-8-sig')
print(df.head(10))

def perform_pairwise_delong(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    results = []

    folder_type = 'Internal' if 'AUC_result' in folder_path else 'External'

    for file1, file2 in combinations(files, 2):
        path1 = os.path.join(folder_path, file1)
        path2 = os.path.join(folder_path, file2)

        d1 = np.load(path1, allow_pickle=True)
        auc1, var1 = get_auc_and_variance(d1['y_true'], d1['y_probs'])

        d2 = np.load(path2, allow_pickle=True)
        auc2, var2 = get_auc_and_variance(d2['y_true'], d2['y_probs'])

        diff = auc1 - auc2
        se = np.sqrt(var1 + var2)

        if se == 0:
            z = 0
            p_value = 1.0
        else:
            z = diff / se
            p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))

        results.append({
            'Type': folder_type,
            'Model_A': file1.replace('.npz', ''),
            'Model_B': file2.replace('.npz', ''),
            'AUC_A': round(auc1, 4),
            'AUC_B': round(auc2, 4),
            'Diff (A-B)': round(diff, 4),
            'P_Value': f"{p_value:.6f}",
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    return pd.DataFrame(results)

if __name__ == "__main__":
    internal_dir = 'AUC_result'
    external_dir = 'AUC_exteral_result'

    all_comparison_results = []

    if os.path.exists(internal_dir):
        print(f"Processing : {internal_dir}...")
        df_internal = perform_pairwise_delong(internal_dir)
        all_comparison_results.append(df_internal)
    else:
        print(f"Directory not found : {internal_dir}")

    if os.path.exists(external_dir):
        print(f"Processing : {external_dir}...")
        df_external = perform_pairwise_delong(external_dir)
        all_comparison_results.append(df_external)
    else:
        print(f"Directory not found : {external_dir}")

    if all_comparison_results:
        final_df = pd.concat(all_comparison_results, ignore_index=True)

        print("\n--- DeLong Test Pairwise comparison results ---")
        print(final_df.to_string())

        final_df.to_csv("./Statistical analysis/delong_test_summary.csv", index=False, encoding='utf-8-sig')
        print("\nThe results have been saved to delong_test_summary.csv")
