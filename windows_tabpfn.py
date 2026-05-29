import pandas as pd
import numpy as np
import math
df = pd.read_csv("./cohort_data.csv", index_col=0)
df.drop(['n_52_0_0', 's_40000_0_0', 's_90010_0_0', 's_90011_0_0'], axis=1, inplace=True, errors='ignore')

year_column = 'dementia_followyear'
df[year_column] = df[year_column].fillna(0)

max_year = int(math.ceil(df[year_column].max()))
print(f"最大随访时长: {df[year_column].max():.2f} 年，将生成 {max_year} 个年度指示列。")

ordinal_mapping = {
    1: 'first', 2: 'second', 3: 'third', 4: 'fourth', 5: 'fifth',
    6: 'sixth', 7: 'seventh', 8: 'eighth', 9: 'ninth', 10: 'tenth'
}

for year in range(1, max_year + 1):
    col_name = ordinal_mapping.get(year, f"year_{year}")
    df[col_name] = ((df[year_column] > 0) & (df[year_column] <= year)).astype(int)
print("\n数据处理完成！")
example_check = df[df[year_column].between(1.9, 2.2)][[year_column, 'first', 'second', 'third']].head()
if not example_check.empty:
    print("精确度检查（发病时间在 2.1 年附近的样本）：")
    print(example_check)
print(f"\n最终数据框形状: {df.shape}")

df.drop(['dementia_followyear'],axis=1,inplace=True)

import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
import os
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, accuracy_score, precision_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings('ignore')

acce_features_data = pd.read_csv("./new_accelerometer_feature_results.csv")
# acce_features_data.drop(['filename','total_rows'],axis=1,inplace=True)
acce_features_data.drop(['filename'],axis=1,inplace=True)
acce_features_data.rename(columns={'eid': 'n_eid'}, inplace=True)
# 处理所有可能包含 np.int64 的列
for col in acce_features_data.columns:
    # 检查列中是否有 np.int64 格式的数据
    if acce_features_data[col].astype(str).str.contains('np\.int64').any():
        # 提取数字
        acce_features_data[col] = acce_features_data[col].astype(str).str.extract(r'np\.int64\((\d+)\)').astype(float)
acce_features_data.dropna(inplace=True)
acce_features_data = acce_features_data.drop_duplicates(subset=['n_eid'])

cohort_data = pd.read_csv("./df_cleaned.csv")

sup1 = pd.read_sas("./id6150_6155.sas7bdat", format='sas7bdat')
target_cols = ['n_6150_0_0', 'n_6150_0_1', 'n_6150_0_2', 'n_6150_0_3']
sup1['angina'] = (sup1[target_cols] == 2).any(axis=1).astype(int)

sup2 = pd.read_csv("./sup_data.csv")
sup_data = pd.merge(sup1, sup2, on="n_eid", how="inner")
sup_data = sup_data[['n_eid','angina','Multivitamins']]

merged_data = pd.merge(cohort_data, sup_data, on="n_eid", how="inner")
status_columns = [
    'self_employed', 'voluntary', 'student', 'lookingafter_home',
    'unemployed', 'retired', 'sickness'
]
# 优先级定义 (数值越小优先级越高)
priority = {
    'self_employed': 1,
    'voluntary': 2,
    'student': 3,
    'lookingafter_home': 4,
    'unemployed': 5,
    'retired': 6,
    'sickness': 7
}
def merge_employment_status(row):
    active_statuses = [col for col in status_columns if row[col] == 1]
    if not active_statuses:
        return np.nan # 或者返回 'missing'
    best_status = min(active_statuses, key=lambda x: priority[x])
    return priority[best_status]
# 执行合并逻辑
# 痴呆状态列名为 'dementia' (0=健康, 1=痴呆)
merged_data['employment_status'] = merged_data.apply(merge_employment_status, axis=1)
merged_data.drop(["self_employed","voluntary","student","lookingafter_home","unemployed","retired","sickness"],axis=1,inplace=True)

status_columns = [
    'college', 'NVQ', 'A_levels', 'O_levels',
    'CSEs', 'Other_Qualifications']
# 优先级定义 (数值越小优先级越高)
priority = {
    'college': 1,
    'NVQ': 2,
    'A_levels': 3,
    'O_levels': 4,
    'CSEs': 5,
    'Other_Qualifications': 6,
}
def merge_employment_status(row):
    active_statuses = [col for col in status_columns if row[col] == 1]
    if not active_statuses:
        return np.nan # 或者返回 'missing'
    best_status = min(active_statuses, key=lambda x: priority[x])
    return priority[best_status]
# 执行合并逻辑
# 痴呆状态列名为 'dementia' (0=健康, 1=痴呆)
merged_data['qualifications_status'] = merged_data.apply(merge_employment_status, axis=1)
merged_data.drop(["college","NVQ","A_levels","O_levels","CSEs","Other_Qualifications"],axis=1,inplace=True)


# 使用 pandas 读取 SAS 文件
assessment_center = pd.read_sas("./assessment_center.sas7bdat", format='sas7bdat')
assessment_center['n_54_0_0'] = assessment_center['n_54_0_0'].astype('int')
assessment_center['n_eid'] = assessment_center['n_eid'].astype('int')
print(f"数据形状: {assessment_center.shape}")
coding10 = pd.read_csv("./coding10.tsv", sep='\t')
print(f"数据形状: {coding10.shape}")

address_df = pd.merge(assessment_center , coding10 , left_on='n_54_0_0',right_on='coding',how='inner')
address_df.drop(["n_54_0_0","coding"],axis=1,inplace=True)

df_windows = df[['n_eid', 'second', 'fourth', 'sixth']]
df_cohort_windows = pd.merge(cohort_data, df_windows, on="n_eid", how="inner")
df_cohort_windows['n_eid'] = df_cohort_windows['n_eid'].astype('int')

address_cohort_df = pd.merge(address_df , df_cohort_windows , on="n_eid", how="inner")

region_mapping = {
    # England (英格兰)
    'Bristol': 'England', 'Leeds': 'England', 'Reading': 'England', 
    'Nottingham': 'England', 'Hounslow': 'England', 'Croydon': 'England',
    'Newcastle': 'England', 'Sheffield': 'England', 'Liverpool': 'England',
    'Birmingham': 'England', 'Bury': 'England', 'Middlesbrough': 'England',
    'Oxford': 'England', 'Stoke': 'England', 'Manchester': 'England',
    'Stockport (pilot)': 'England', 'Barts': 'England',
    
    # Scottish (苏格兰)
    'Edinburgh': 'Scottish', 'Glasgow': 'Scottish',
    
    # Wales (威尔士)
    'Cardiff': 'Wales', 'Swansea': 'Wales', 'Wrexham': 'Wales'
}
cutoff_dates = {
    'England': '2023-03-31',
    'Scottish': '2022-08-31', 
    'Wales': '2022-05-31'
}
def map_city_to_region(city):
    if pd.isna(city):
        return 'Unknown'
    city = str(city).strip()
    for city_name, region in region_mapping.items():
        if city_name.lower() in city.lower():
            return region
    return 'Unknown'
# 添加地区列
address_cohort_df['region'] = address_cohort_df['meaning'].apply(map_city_to_region)

temp_sup_features = merged_data[['n_eid', 'employment_status', 'qualifications_status', 'angina', 'Multivitamins']]
df_cohort = pd.merge(address_cohort_df, temp_sup_features, on='n_eid', how='inner')
merged_data = pd.merge(df_cohort, acce_features_data, on='n_eid', how='inner')
wearable_cols = [c for c in acce_features_data.columns if c not in ['n_eid', 'total_rows', 'filename']]
primary_cols = [c for c in cohort_data.columns if c not in ['n_eid', 'dementia']] # 排除ID和原始标签
print(merged_data.shape)
print(merged_data['dementia'].value_counts())

england_df = merged_data[merged_data['region'] == "England"].copy()
external_df = merged_data[merged_data['region'].isin(["Scottish", "Wales"])].copy()
top5 = [
    'n_21022_0_0', 'illness', 'n_23099_0_0', 'paired_correct', 'eyeproblem'
]
top10 = [
    'n_21022_0_0', 'illness', 'n_23099_0_0', 'paired_correct', 'eyeproblem',
    'n_137_0_0', 'employment_status', 'financial_difficulty', 'paired_time', 'sunlamp_use'
]
top15 = [
    'n_21022_0_0', 'illness', 'n_23099_0_0', 'paired_correct', 'eyeproblem',
    'n_137_0_0', 'employment_status', 'financial_difficulty', 'paired_time', 'sunlamp_use',
    'mood_swings', 'malzheimers', 'diabetes', 'drink_meal', 'n_31_0_0'
]
top20 = [
    'n_21022_0_0', 'illness', 'n_23099_0_0', 'paired_correct', 'eyeproblem',
    'n_137_0_0', 'employment_status', 'financial_difficulty', 'paired_time', 'sunlamp_use',
    'mood_swings', 'malzheimers', 'diabetes', 'drink_meal', 'n_31_0_0',
    'fall', 'laxatives', 'malive', 'DVT', 'n_21001_0_0'
]

models_config = {
   "Only wearable": wearable_cols,
   "Only all primary": primary_cols,
    "Top1 + wearable": ['n_21022_0_0'] + wearable_cols,
   "Top5 + wearable": top5 + wearable_cols,
   "Top10 + wearable": top10 + wearable_cols,
   "Top15 + wearable": top15 + wearable_cols,
   "Top20 + wearable": top20 + wearable_cols,
   "All + wearable": primary_cols + wearable_cols
}

window_config = {
    "0-2 years": "second",
    "0-4 years": "fourth",
    "0-6 years": "sixth"
}

def impute_and_scale(train_X, test_X):
    train_X, test_X = train_X.copy(), test_X.copy()
    
    # 识别列类型
    num_cols = train_X.select_dtypes(include='number').columns
    cat_cols = train_X.select_dtypes(include=['object', 'category']).columns
    
    # 填充数值列
    if not num_cols.empty:
        fill_num = train_X[num_cols].median()
        train_X[num_cols] = train_X[num_cols].fillna(fill_num)
        test_X[num_cols] = test_X[num_cols].fillna(fill_num)
        
    # 填充分类列
    if not cat_cols.empty:
        fill_cat = train_X[cat_cols].mode().iloc[0]
        train_X[cat_cols] = train_X[cat_cols].fillna(fill_cat)
        test_X[cat_cols] = test_X[cat_cols].fillna(fill_cat)
        
    # 标准化
    if not num_cols.empty:
        scaler = StandardScaler()
        train_X[num_cols] = scaler.fit_transform(train_X[num_cols])
        test_X[num_cols] = scaler.transform(test_X[num_cols])
        
    # 转哑变量
    train_X = pd.get_dummies(train_X)
    test_X = pd.get_dummies(test_X).reindex(columns=train_X.columns, fill_value=0)
    
    return train_X, test_X

def calculate_all_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0)
    }

internal_output_dir = "./AUC_results_internal_England"
external_output_dir = "./AUC_results_external_Validation"
for d in [internal_output_dir, external_output_dir]:
    if not os.path.exists(d): os.makedirs(d)

model = TabPFNClassifier(model_path="./tabpfn_models/tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                ignore_pretraining_limits=True,
                                device='cuda')
for window_name, target_col in window_config.items():
    print(f"\n{'='*30} 窗口: {window_name} ({target_col}) {'='*30}")

    for model_label, feature_list in models_config.items():
        print(f"\n>>> 正在执行: {model_label}")

        current_data_all = merged_data[['n_eid', 'region', target_col] + feature_list].dropna(subset=[target_col])

        internal_df = current_data_all[current_data_all['region'] == 'England']
        external_df = current_data_all[current_data_all['region'].isin(['Scottish', 'Wales'])]
        
        X_internal = internal_df[feature_list]
        y_internal = internal_df[target_col]
        X_external_raw = external_df[feature_list]
        y_external = external_df[target_col]

        iter_metrics_internal = []
        iter_metrics_external = []
        
        # 存储预测概率用于保存 npz
        all_y_true_int, all_y_probs_int = [], []
        all_y_true_ext, all_y_probs_ext = [], []

        # 执行 5 次随机切分 (8:2) 以评估稳定性
        for i in range(5):
            # 仅对英格兰数据进行切分
            X_train_full, X_test_raw, y_train_full, y_test_int = train_test_split(
                X_internal, y_internal, test_size=0.2, stratify=y_internal, random_state=i*42
            )

            # 内部测试集标准化
            X_train_final, X_test_final = impute_and_scale(X_train_full, X_test_raw)
            # 外部验证集标准化 (同样使用 X_train_full 的参数)
            _, X_external_final = impute_and_scale(X_train_full, X_external_raw)

            sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train_final, y_train_full)

            model.fit(X_res, y_res)

            probs_int = model.predict_proba(X_test_final)[:, 1]
            preds_int = model.predict(X_test_final)
            iter_metrics_internal.append(calculate_all_metrics(y_test_int, preds_int, probs_int))
            all_y_true_int.append(y_test_int.values)
            all_y_probs_int.append(probs_int)

            probs_ext = model.predict_proba(X_external_final)[:, 1]
            preds_ext = model.predict(X_external_final)
            iter_metrics_external.append(calculate_all_metrics(y_external, preds_ext, probs_ext))
            all_y_true_ext.append(y_external.values)
            all_y_probs_ext.append(probs_ext)

        def get_summary_df(metrics_list):
            df = pd.DataFrame(metrics_list)
            means = df.mean()
            stds = df.std()
            return pd.DataFrame({
                'Mean ± Std': [f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)]
            }, index=means.index)

        res_int_summary = get_summary_df(iter_metrics_internal)
        res_ext_summary = get_summary_df(iter_metrics_external)

        print(f"\n[{model_label} | {window_name}] 性能展示:")
        # 合并内外结果展示对比
        comparison_df = pd.concat([res_int_summary, res_ext_summary], axis=1)
        comparison_df.columns = ['Internal (England)', 'External (Scot/Wales)']
        print(comparison_df)

        # --- 保存结果文件 ---
        file_name = f"{model_label.replace(' ', '_')}_{window_name.replace(' ', '_')}.npz"
        
        np.savez(os.path.join(internal_output_dir, file_name), 
                 y_true=np.array(all_y_true_int, dtype=object), 
                 y_probs=np.array(all_y_probs_int, dtype=object))
        
        np.savez(os.path.join(external_output_dir, file_name), 
                 y_true=np.array(all_y_true_ext, dtype=object), 
                 y_probs=np.array(all_y_probs_ext, dtype=object))

print("\n所有窗口的内外部验证及指标汇总已完成。")


