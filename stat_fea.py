import pandas as pd
import os
import re
import numpy as np
from scipy.stats import skew, kurtosis

INPUT_DIR = "/data_hou/wangxinyu/accelerometer_data/processed_acc_data/"
OUTPUT_PATH = "./new_accelerometer_feature_results.csv"
MIN_ROWS = 5760  # one day
DAY_START_HOUR = 6  # （6:00）
DAY_END_HOUR = 22  #（22:00）
SEDENTARY_THRESHOLD = 60

NUMERIC_COLS = ["acc", "MET", "light", "moderate", "vigorous"]
ACTIVITY_COLS = ["sedentary", "sleep", "light", "moderate-vigorous", "CpSB", "CpLPA", "CpMPA", "CpVPA", "CpMVPA"]

def extract_eid_from_filename(filename):
    match = re.match(r'^(\d{7})', filename)
    if match:
        return match.group(1)
    else:
        return None


def filter_file_by_length(df):
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
    features = {}

    for col in NUMERIC_COLS:
        if col not in df.columns:
            continue
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue

        features[f"{col}_mean"] = col_data.mean()
        features[f"{col}_median"] = col_data.median()

        features[f"{col}_std"] = col_data.std()
        features[f"{col}_var"] = col_data.var()
        features[f"{col}_iqr"] = col_data.quantile(0.75) - col_data.quantile(0.25)

        features[f"{col}_skew"] = skew(col_data)
        features[f"{col}_kurtosis"] = kurtosis(col_data)
        features[f"{col}_25pct"] = col_data.quantile(0.25)
        features[f"{col}_75pct"] = col_data.quantile(0.75)
        features[f"{col}_90pct"] = col_data.quantile(0.90)
        features[f"{col}_max"] = col_data.max()
        features[f"{col}_min"] = col_data.min()
        features[f"{col}_range"] = col_data.max() - col_data.min()

    total_rows = len(df)
    for col in ACTIVITY_COLS:
        if col not in df.columns:
            continue
        activity_count = df[col].sum()
        features[f"{col}_ratio"] = activity_count / total_rows
    
    return features


def calculate_daynight_met_diff(df):
    features = {}

    daytime_met = df[df["is_daytime"] == 1]["MET"].mean()

    nighttime_met = df[df["is_daytime"] == 0]["MET"].mean()

    if nighttime_met > 0:
        features["daynight_met_ratio"] = daytime_met / nighttime_met
    else:
        features["daynight_met_ratio"] = np.nan

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
    return slope

def calculate_sleep_duration(df):
    if "sleep" not in df.columns:
        return {"sleep_duration": np.nan}
    sleep_data = df["sleep"].astype(int)
    sleep_duration = sleep_data.sum() * 30
    return {"sleep_duration": sleep_duration}

def calculate_sedentary_duration(df):
    if "sedentary" not in df.columns:
        return {"sedentary_duration": np.nan}
    sedentary_data = df["sedentary"].astype(int)
    sedentary_duration = sedentary_data.sum() * 30
    return {"sedentary_duration": sedentary_duration}

def calculate_activity_duration(df, activity_col):
    if activity_col not in df.columns:
        return {f"{activity_col}_duration": np.nan}
    activity_data = df[activity_col].astype(int)
    activity_duration = activity_data.sum() * 30
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
    all_features = []
    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]
    
    print(f"Found {len(csv_files)} csv files，Start processing...")
    for idx, filename in enumerate(csv_files, 1):
        try:
            eid = extract_eid_from_filename(filename)
            if not eid:
                print(f"Skip file {filename}：Unable to extract eid")
                continue

            file_path = os.path.join(INPUT_DIR, filename)
            df = pd.read_csv(file_path)

            if not filter_file_by_length(df):
                print(f"Skip file {filename}（eid:{eid}）：Number of data rows {len(df)}<{MIN_ROWS}（1 day）")
                continue

            df = process_time_and_daynight(df)

            basic_features = calculate_basic_stat_features(df)
            daynight_features = calculate_daynight_met_diff(df)
            sedentary_features = calculate_sedentary_streaks(df)

            autocorrelation_features = {f"acc_autocorr_{lag}": calculate_autocorrelation(df, "acc", lag) for lag in [1, 5, 10]}
            trend_features = {f"acc_trend": calculate_trend(df, "acc")}
            sleep_duration_features = calculate_sleep_duration(df)
            sedentary_duration_features = calculate_sedentary_duration(df)
            activity_duration_features = {f"{col}_duration": calculate_activity_duration(df, col) for col in ACTIVITY_COLS}
            max_slope_features = {f"acc_max_slope": calculate_max_slope(df, "acc")}

            combined_features = {
                "eid": eid,
                "filename": filename,
                "total_rows": len(df)
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
                print(f"Processed {idx}/{len(csv_files)} files，Cumulative valid documents {len(all_features)}")
        
        except Exception as e:
            print(f"Processing files {filename} error：{str(e)}，Skip this file")
            continue

    if all_features:
        result_df = pd.DataFrame(all_features)
        result_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
        print(f"\nProcessing complete! A total of {len(csv_files)} files were processed, with {len(all_features)} valid files.")
        print(f"Feature results have been saved to: {OUTPUT_PATH}")
        # Preview of the first 5 lines
        print("\nPreview of feature results (first 5 lines):")
        print(result_df.head())
    else:
        print("\nProcessing complete, but no valid files matching the criteria were found (data rows ≥ 2880)")

if __name__ == "__main__":
    main()

