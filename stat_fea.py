import pandas as pd
import os
import re
import numpy as np
from scipy.stats import skew, kurtosis  # 用于计算偏度、峰度

INPUT_DIR = "/data_hou/wangxinyu/accelerometer_data/processed_acc_data/"
OUTPUT_PATH = "./new_accelerometer_feature_results.csv"
MIN_ROWS = 5760  # 1天（30秒采样）所需最小数据行数
DAY_START_HOUR = 6  # 白天起始小时（6:00）
DAY_END_HOUR = 22  # 白天结束小时（22:00）
SEDENTARY_THRESHOLD = 60  # 久坐超30分钟所需连续点数（30分钟/30秒=60点）

NUMERIC_COLS = ["acc", "MET", "light", "moderate", "vigorous"]
# 活动类型列（用于计算活动占比）
ACTIVITY_COLS = ["sedentary", "sleep", "light", "moderate-vigorous", "CpSB", "CpLPA", "CpMPA", "CpVPA", "CpMVPA"]

def extract_eid_from_filename(filename):
    """从文件名提取前7位eid"""
    match = re.match(r'^(\d{7})', filename)  # 匹配开头7位数字
    if match:
        return match.group(1)
    else:
        return None  # 若无法提取eid，标记为None


def filter_file_by_length(df):
    """根据数据行数筛选文件（≥2880行为1天）"""
    return len(df) >= MIN_ROWS


def process_time_and_daynight(df):
    df["time_clean"] = df["time"].str.replace(r'\s*\[[^\]]+\]$', '', regex=True)

    time_format = "%Y-%m-%d %H:%M:%S.%f%z"
    df["time_parsed"] = pd.to_datetime(
        df["time_clean"], 
        format=time_format
        utc=True
    ).dt.tz_localize(None)

    df["hour"] = df["time_parsed"].dt.hour
    df["is_daytime"] = ((df["hour"] >= DAY_START_HOUR) & (df["hour"] < DAY_END_HOUR)).astype(int)
    return df


def calculate_basic_stat_features(df):
    """计算基础时域统计特征"""
    features = {}

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue  # 若列不存在，跳过
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        
        # 集中趋势
        features[f"{col}_mean"] = col_data.mean()
        features[f"{col}_median"] = col_data.median()
        # 离散程度
        features[f"{col}_std"] = col_data.std()
        features[f"{col}_var"] = col_data.var()
        features[f"{col}_iqr"] = col_data.quantile(0.75) - col_data.quantile(0.25)
        # 分布特征
        features[f"{col}_skew"] = skew(col_data)  # 偏度
        features[f"{col}_kurtosis"] = kurtosis(col_data)  # 峰度
        features[f"{col}_25pct"] = col_data.quantile(0.25)
        features[f"{col}_75pct"] = col_data.quantile(0.75)
        features[f"{col}_90pct"] = col_data.quantile(0.90)
        # 极值特征
        features[f"{col}_max"] = col_data.max()
        features[f"{col}_min"] = col_data.min()
        features[f"{col}_range"] = col_data.max() - col_data.min()

    total_rows = len(df)
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            continue
        activity_count = df[col].sum()  # 活动为1的次数
        features[f"{col}_ratio"] = activity_count / total_rows  # 时间占比
    
    return features


def calculate_daynight_met_diff(df):
    features = {}
    # 白天MET均值
    daytime_met = df[df["is_daytime"] == 1]["MET"].mean()
    # 夜间MET均值
    nighttime_met = df[df["is_daytime"] == 0]["MET"].mean()
    
    # 昼夜MET均值比（避免夜间均值为0导致除以0）
    if nighttime_met > 0:
        features["daynight_met_ratio"] = daytime_met / nighttime_met
    else:
        features["daynight_met_ratio"] = np.nan  # 夜间无数据时标记为NaN
    
    # 额外补充昼夜MET绝对值差异
    features["daynight_met_diff"] = daytime_met - nighttime_met
    return features


def calculate_sedentary_streaks(df):
    if "sedentary" not in df.columns:
        return {"sedentary_30min_streak_count": np.nan}
    
    sedentary = df["sedentary"].astype(int)
    streak_starts = (sedentary == 1) & (sedentary.shift(1) == 0)
    streak_ids = streak_starts.cumsum() * sedentary
    streak_lengths = streak_ids[streak_ids > 0].value_counts().values
    
    long_streak_count = sum(length >= SEDENTARY_THRESHOLD for length in streak_lengths)
    return {"sedentary_30min_streak_count": long_streak_count}


def calculate_autocorrelation(df, col, lag=1):
    col_data = df[col].dropna()
    if len(col_data) <= lag:
        return np.nan
    autocorr = col_data.autocorr(lag=lag)
    return autocorr

def calculate_trend(df, col):
    col_data = df[col].dropna()
    if len(col_data) < 2:
        return np.nan
    x = np.arange(len(col_data))
    y = col_data.values
    slope, intercept = np.polyfit(x, y, 1)
    return slope  # 返回斜率作为趋势

def calculate_sleep_duration(df):
    if "sleep" not in df.columns:
        return {"sleep_duration": np.nan}
    sleep_data = df["sleep"].astype(int)
    sleep_duration = sleep_data.sum() * 30  # 每个点代表30秒
    return {"sleep_duration": sleep_duration}

def calculate_sedentary_duration(df):
    if "sedentary" not in df.columns:
        return {"sedentary_duration": np.nan}
    sedentary_data = df["sedentary"].astype(int)
    sedentary_duration = sedentary_data.sum() * 30  # 每个点代表30秒
    return {"sedentary_duration": sedentary_duration}

def calculate_activity_duration(df, activity_col):
    if activity_col not in df.columns:
        return {f"{activity_col}_duration": np.nan}
    activity_data = df[activity_col].astype(int)
    activity_duration = activity_data.sum() * 30  # 每个点代表30秒
    return {f"{activity_col}_duration": activity_duration}

def calculate_max_slope(df, col):
    col_data = df[col].dropna()
    if len(col_data) < 2:
        return np.nan
    x = np.arange(len(col_data))
    y = col_data.values
    slopes = np.diff(y) / np.diff(x)
    return np.max(np.abs(slopes))

def main():
    # 初始化结果列表（存储每个eid的特征）
    all_features = []
    # 获取目录下所有CSV文件
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    
    print(f"发现{len(csv_files)}个CSV文件，开始处理...")
    for idx, filename in enumerate(csv_files, 1):
        try:
            eid = extract_eid_from_filename(filename)
            if not eid:
                print(f"跳过文件{filename}：无法提取eid")
                continue

            file_path = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(file_path)

            if not filter_file_by_length(df):
                print(f"跳过文件{filename}（eid:{eid}）：数据行数{len(df)}<{MIN_ROWS}（1天）")
                continue

            df = process_time_and_daynight(df)

            basic_features = calculate_basic_stat_features(df)
            daynight_features = calculate_daynight_met_diff(df)
            sedentary_features = calculate_sedentary_streaks(df)
            # 计算时序和活动相关特征
            autocorrelation_features = {f"acc_autocorr_{lag}": calculate_autocorrelation(df, "acc", lag) for lag in [1, 5, 10]}
            trend_features = {f"acc_trend": calculate_trend(df, "acc")}
            sleep_duration_features = calculate_sleep_duration(df)
            sedentary_duration_features = calculate_sedentary_duration(df)
            activity_duration_features = {f"{col}_duration": calculate_activity_duration(df, col) for col in ACTIVITY_COLS}
            max_slope_features = {f"acc_max_slope": calculate_max_slope(df, "acc")}

            combined_features = {
                "eid": eid,
                "filename": filename,
                "total_rows": len(df)  # 记录实际数据行数
            }
            combined_features.update(basic_features)
            combined_features.update(daynight_features)
            combined_features.update(sedentary_features)
            combined_features.update(autocorrelation_features)
            combined_features.update(trend_features)
            combined_features.update(sleep_duration_features)
            combined_features.update(sedentary_duration_features)
            combined_features.update(activity_duration_features)
            combined_features.update(max_slope_features)

            all_features.append(combined_features)

            if idx % 100 == 0:
                print(f"已处理{idx}/{len(csv_files)}个文件，累计有效文件{len(all_features)}个")
        
        except Exception as e:
            print(f"处理文件{filename}时出错：{str(e)}，跳过该文件")
            continue

    if all_features:
        # 转换为DataFrame
        result_df = pd.DataFrame(all_features)
        # 保存为CSV
        result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"\n处理完成！共处理{len(csv_files)}个文件，有效文件{len(all_features)}个")
        print(f"特征结果已保存至：{OUTPUT_PATH}")
        # 打印前5行预览
        print("\n特征结果预览（前5行）：")
        print(result_df.head())
    else:
        print("\n处理完成，但未找到符合条件的有效文件（数据行数≥2880）")

if __name__ == "__main__":
    main()

