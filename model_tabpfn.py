import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
import re
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from tabpfn import TabPFNClassifier
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 全英文无衬线字体
plt.rcParams['axes.unicode_minus'] = False          # 解决负号显示问题
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 100

## 读取年龄文件
age_df = pd.read_csv("./base.csv",index_col=0)
## 读取队列数据
cohort_data = pd.read_csv("./cohort_data.csv",index_col=0)
cohort_data.drop(['n_52_0_0','dementia_followyear','s_40000_0_0','s_90010_0_0'],axis=1,inplace=True)
cohort_data = pd.merge(age_df[['n_eid','n_21022_0_0']],cohort_data,on="n_eid",how="inner")


## 根据20%的缺失比例去除参与者
def remove_high_missing_data(df, threshold=0.3):
    df_cleaned = df.copy()
    print(f"原始数据形状: {df_cleaned.shape}")
    print(f"设定的缺失值阈值: {threshold*100}%")
    print("\n=== 处理特征（列）===")

    missing_ratio_columns = df_cleaned.isnull().mean()

    columns_to_drop = missing_ratio_columns[missing_ratio_columns > threshold].index.tolist()
    
    print(f"需要删除的列数量: {len(columns_to_drop)}")
    if len(columns_to_drop) > 0:
        print(f"要删除的列名: {columns_to_drop}")
        for col in columns_to_drop:
            print(f"  - {col}: {missing_ratio_columns[col]:.2%}")
    
    df_cleaned = df_cleaned.drop(columns=columns_to_drop)
    print(f"删除高缺失值列后的数据形状: {df_cleaned.shape}")
    print("\n=== 处理样本（行）===")
    
    # 计算每行的缺失值比例
    missing_ratio_rows = df_cleaned.isnull().mean(axis=1)
    
    # 找出缺失比例超过阈值的行
    rows_to_drop = missing_ratio_rows[missing_ratio_rows > threshold].index.tolist()
    
    print(f"需要删除的行数量: {len(rows_to_drop)}")
    
    # 删除这些行
    df_cleaned = df_cleaned.drop(index=rows_to_drop)
    print(f"删除高缺失值行后的数据形状: {df_cleaned.shape}")

    print("\n=== 处理摘要 ===")
    original_size = df.shape[0] * df.shape[1]
    final_size = df_cleaned.shape[0] * df_cleaned.shape[1]
    missing_before = df.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    print(f"原始数据点总数: {original_size}")
    print(f"最终数据点总数: {final_size}")
    print(f"删除的数据点比例: {(1 - final_size/original_size):.2%}")
    print(f"处理前缺失值总数: {missing_before}")
    print(f"处理后缺失值总数: {missing_after}")
    print(f"剩余缺失值比例: {missing_after/final_size:.2%}" if final_size > 0 else "数据全部被删除")
    
    return df_cleaned

df_cleaned = remove_high_missing_data(cohort_data, threshold=0.2)

## 读取加速度计衍生数据
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

## 读取招募时间的文件
recruitment_date = pd.read_sas("./recruitment_date.sas7bdat", format='sas7bdat')
recruitment_date['n_eid'] = recruitment_date['n_eid'].astype('int')
print(f"数据形状: {recruitment_date.shape}")

## 校正年龄
cohort_data = df_cleaned
cohort_data = pd.merge(recruitment_date, cohort_data, on='n_eid', how='inner')
cohort_data['s_90011_0_0'] = pd.to_datetime(cohort_data['s_90011_0_0'], errors='coerce')
cohort_data['s_53_0_0'] = pd.to_datetime(cohort_data['s_53_0_0'], errors='coerce')
cohort_data['year_diff'] = (cohort_data['s_90011_0_0'] - cohort_data['s_53_0_0']).dt.total_seconds() / (365.25 * 24 * 3600)
cohort_data['n_21022_0_0'] = (cohort_data['n_21022_0_0'] + cohort_data['year_diff']).round(1)
cohort_data = cohort_data.drop(columns=['year_diff','s_53_0_0','s_90011_0_0'])

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


# merged_data = pd.merge(cohort_data, acce_features_data, on="n_eid", how="inner")
##所有问卷数据
# merged_data = merged_data.iloc[:,:256]

# merged_data = merged_data[top5]
## 可以将不同的数据组合保存下来，然后读取即可
merged_data = pd.read_csv("./data/wearable_top1.csv",index_col=0)

assessment_center = pd.read_sas("./assessment_center.sas7bdat", format='sas7bdat')
assessment_center['n_54_0_0'] = assessment_center['n_54_0_0'].astype('int')
assessment_center['n_eid'] = assessment_center['n_eid'].astype('int')
print(f"数据形状: {assessment_center.shape}")

coding10 = pd.read_csv("./coding10.tsv", sep='\t')
print(f"数据形状: {coding10.shape}")

address_df = pd.merge(assessment_center , coding10 , left_on='n_54_0_0',right_on='coding',how='inner')
address_df.drop(["n_54_0_0","coding"],axis=1,inplace=True)

address_cohort_df = pd.merge(address_df , merged_data , on="n_eid", how="inner")
address_cohort_df

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
    """将城市映射到地区"""
    # 处理可能的NaN值
    if pd.isna(city):
        return 'Unknown'
    city = str(city).strip()
    for city_name, region in region_mapping.items():
        if city_name.lower() in city.lower():
            return region
    return 'Unknown'
# 添加地区列
address_cohort_df['region'] = address_cohort_df['meaning'].apply(map_city_to_region)

## 英格兰数据和外部验证数据分割（英格兰/苏格兰、威尔士）
england_df = address_cohort_df[address_cohort_df['region'] == "England"].copy()
external_df = address_cohort_df[address_cohort_df['region'].isin(["Scottish", "Wales"])].copy()

# 定义不需要参与计算的列
drop_cols = ["n_eid", "meaning", "region"]
def prepare_xy(df):
    X = df.drop(columns=[col for col in drop_cols if col in df.columns] + ['dementia'])
    y = df['dementia']
    return X, y

def impute_mixed_data(train_X, test_X):
    """
    基于 train_X 计算中位数和众数，并应用到 train_X 和 test_X 上
    """
    train_X = train_X.copy()
    test_X = test_X.copy()
    # 识别列类型
    num_cols = train_X.select_dtypes(include='number').columns
    cat_cols = train_X.select_dtypes(include=['object', 'category']).columns
    # 计算训练集的中位数和众数
    num_fill_values = train_X[num_cols].median()
    # 众数可能返回多行，取第一行
    cat_modes = train_X[cat_cols].mode()
    cat_fill_values = cat_modes.iloc[0] if not cat_modes.empty else None
    # 执行填充
    if not num_fill_values.empty:
        train_X[num_cols] = train_X[num_cols].fillna(num_fill_values)
        test_X[num_cols] = test_X[num_cols].fillna(num_fill_values)
    if cat_fill_values is not None:
        train_X[cat_cols] = train_X[cat_cols].fillna(cat_fill_values)
        test_X[cat_cols] = test_X[cat_cols].fillna(cat_fill_values)
    return train_X, test_X

def calculate_metrics(y_true, y_pred, y_prob, ty_val = 'interal'):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    model_filename1 = "./AUC_result/Wearable-derived features + top 1 selected variables.npz" #根据实际情况进行调整
    model_filename2 = "./AUC_exteral_result/Wearable-derived features + top 1 selected variables.npz"
    if ty_val == 'internal':
        np.savez(model_filename1, 
             y_true=np.array(y_true), 
             y_probs=np.array(y_prob))
    elif ty_val == 'external':
        np.savez(model_filename2, 
             y_true=np.array(y_true), 
             y_probs=np.array(y_prob))
    return {
        "AUC": roc_auc_score(y_true, y_prob),
        "Recall": recall_score(y_true, y_pred),
        "Sensitivity": recall_score(y_true, y_pred),
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0)
    }

iterations = 5
results_test = []
results_external = []

for i in range(iterations):
    print(f"\n>>> 开始第 {i+1}/5 次随机划分评估...")
    
    # 8:2 划分英格兰数据
    train_df, test_df = train_test_split(
        england_df, test_size=0.2, stratify=england_df['dementia'], random_state=i*42
    )
    
    X_train_full, y_train_full = prepare_xy(train_df)
    X_test_raw, y_test = prepare_xy(test_df)
    X_external_raw, y_external = prepare_xy(external_df)

    # --- 5折交叉验证 (在 80% 训练集内部，仅评估AUC) ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_full, y_train_full), 1):
        X_cv_train_raw = X_train_full.iloc[train_idx]
        X_cv_val_raw = X_train_full.iloc[val_idx]
        y_cv_train, y_cv_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]
        
        # 内部填充与标准化
        X_cv_train, X_cv_val = impute_mixed_data(X_cv_train_raw, X_cv_val_raw)
        num_cols = X_cv_train.select_dtypes(include='number').columns
        scaler = StandardScaler()
        X_cv_train[num_cols] = scaler.fit_transform(X_cv_train[num_cols])
        X_cv_val[num_cols] = scaler.transform(X_cv_val[num_cols])
        
        X_cv_train_final = pd.get_dummies(X_cv_train)
        X_cv_val_final = pd.get_dummies(X_cv_val).reindex(columns=X_cv_train_final.columns, fill_value=0)
        
        sampler = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_res, y_res = sampler.fit_resample(X_cv_train_final, y_cv_train)
        
        model = TabPFNClassifier(model_path="./tabpfn_models/tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                ignore_pretraining_limits=True,
                                device='cuda')
        model.fit(X_res, y_res)
        
        prob = model.predict_proba(X_cv_val_final)[:, 1]
        cv_aucs.append(roc_auc_score(y_cv_val, prob))
        
    print(f"   CV 平均 AUC: {np.mean(cv_aucs):.4f} + {np.std(cv_aucs):.4f}")

    # --- 训练最终模型并评估 ---
    # 填充外部数据 (基于当前轮次 80% 的英格兰训练集)
    X_train_imp, X_test_imp = impute_mixed_data(X_train_full, X_test_raw)
    _, X_ext_imp = impute_mixed_data(X_train_full, X_external_raw)
    
    # 标准化
    num_cols = X_train_imp.select_dtypes(include='number').columns
    final_scaler = StandardScaler()
    X_train_imp[num_cols] = final_scaler.fit_transform(X_train_imp[num_cols])
    X_test_imp[num_cols] = final_scaler.transform(X_test_imp[num_cols])
    X_ext_imp[num_cols] = final_scaler.transform(X_ext_imp[num_cols])
    
    # 转哑变量并对齐列
    X_train_f = pd.get_dummies(X_train_imp)
    def align_cols(df, reference_df):
        return pd.get_dummies(df).reindex(columns=reference_df.columns, fill_value=0)
    
    X_test_f = align_cols(X_test_imp, X_train_f)
    X_ext_f = align_cols(X_ext_imp, X_train_f)
    
    # 下采样并训练
    sampler_f = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
    X_res_f, y_res_f = sampler_f.fit_resample(X_train_f, y_train_full)
    
    final_model = TabPFNClassifier(model_path="./tabpfn_models/tabpfn-v2.5-classifier-v2.5_default.ckpt",
                                  ignore_pretraining_limits=True,
                                  device='cuda')
    final_model.fit(X_res_f, y_res_f)
    
# 评估
    def get_eval(X, y, model, ty_val = 'internal'):
        p = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        return calculate_metrics(y, pred, p, ty_val)

    results_test.append(get_eval(X_test_f, y_test, final_model, ty_val = 'internal'))
    results_external.append(get_eval(X_ext_f, y_external, final_model, ty_val = 'external'))

def show_final(res_list, name):
    # 将包含多个字典的列表转换为 DataFrame
    df = pd.DataFrame(res_list)
    # 计算均值和标准差
    means = df.mean()
    stds = df.std()
    # 构建一个包含 "均值 ± 标准差" 字符串的新 DataFrame
    # 这里保留 4 位小数
    summary_df = pd.DataFrame({
        'Mean ± Std': [f"{m:.4f} ± {s:.4f}" for m, s in zip(means, stds)]
    }, index=means.index)
    print(f"\n=== {name} 最终评估 (5次随机划分汇总) ===")
    print(summary_df)
    print("-" * 40)
show_final(results_test, "英格兰测试集 (20% Test)")
show_final(results_external, "外部验证")
