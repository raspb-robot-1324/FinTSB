import os
import pickle
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.config import REG_CN
from tqdm import tqdm
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, acf
from scipy.signal import periodogram
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# 1. CONFIGURATION (Match FinTSB Paper)
# ==========================================
# Point this to your Qlib data directory
QLIB_DATA_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data")
TEMP_DIR = "./data/FinTSB_cn/temp"  # Temporary storage for intermediate files
OUTPUT_DIR = "./data/FinTSB_cn"       # Final output directory for datasets

# FinTSB Methodology Parameters
SEGMENT_LENGTH = 250  # 1 year of trading days
HISTORY_YEARS = 15    # The paper uses 15 years
TOP_K = 300           # Top/Bottom 300 stocks selection

# Output Categories matching FinTSB folder structure
CATEGORIES = {
    "rise": "uptrend",
    "fall": "downtrend",
    "fluctuation": "oscillatory",
    "extreme": "black_swan"
}

MAD_THRESHOLD = 8.0

def initialize_qlib():
    if not os.path.exists(QLIB_DATA_DIR):
        raise FileNotFoundError(f"Qlib data not found at {QLIB_DATA_DIR}. Please download it first.")
    qlib.init(provider_uri=QLIB_DATA_DIR, region=REG_CN)
    
def remove_temp_dir(dir: str):
    if os.path.exists(dir):
        os.system(f"rm -rf {dir}")

def get_market_data(start_date="2013-01-01", end_date="2023-01-01"):
    """
    Fetches daily OHLCV + Returns for the entire A-Share market.
    """
    print(f"Fetching data from {start_date} to {end_date}...")
    
    # We load 'feature' fields expected by FinTSB models (Open, Close, High, Low, Volume)
    # Ref($close, -1) is used to calculate returns
    fields = ["$open", "$high", "$low", "$close", "$volume", "Ref($close, 1)/$close - 1"]
    names = ["open", "high", "low", "close", "volume", "label"]
    
    # Load all A-shares (market='all') to filter later
    df = D.features(D.instruments(market='all'), fields, start_time=start_date, end_time=end_date)
    df.columns = names
    
    # Drop NaNs (stocks not listed yet)
    df = df.dropna()
    
    # Create the 'return' column for sorting regimes
    # (Close - PrevClose) / PrevClose
    df['return'] = df['close'].pct_change().fillna(0)
    
    return df

def calculate_mad_score(series):
    """
    Calculates the Robust Z-Score (MAD Score).
    """
    median = series.median()
    # Calculate absolute deviation from median
    abs_deviation = np.abs(series - median)
    # Calculate MAD (Median of those deviations)
    mad = abs_deviation.median()
    
    if mad == 0:
        return np.zeros(len(series)) # Avoid division by zero
    
    # 0.6745 is the consistency constant for normal distribution
    robust_z = 0.6745 * (series - median) / mad
    
    return robust_z

def count_positive(series):
    """
    Counts the number of positive returns in the series.
    """
    return (series > 0).sum() / len(series)

def segment_and_save(df):
    """
    Implements the FinTSB splitting logic:
    1. Divide history into non-overlapping 250-day segments.
    2. Classify stocks into regimes.
    3. Save as .pkl
    """
    dates = df.index.get_level_values('datetime').unique().sort_values()
    
    # Create output directories
    for cat in CATEGORIES.keys():
        os.makedirs(os.path.join(TEMP_DIR, cat), exist_ok=True)

    # Process in chunks of SEGMENT_LENGTH (approx 1 year)
    num_segments = len(dates) // SEGMENT_LENGTH
    
    print(f"Processing {num_segments} segments (Years)...")

    for i in tqdm(range(num_segments)):
        # Define time window
        start_idx = i * SEGMENT_LENGTH
        end_idx = (i + 1) * SEGMENT_LENGTH
        segment_dates = dates[start_idx:end_idx]
        
        # Slice DataFrame
        segment_df = df.loc[(slice(None), segment_dates), :]
        
        # Normalize across stocks within this segment to identify regimes
        segment_df = segment_df.groupby('datetime').transform(lambda x: (x - x.mean()) / x.std()).fillna(0)  # Z-score normalization by date
                
        # We group by instrument to normalize each stock individually
        # (Or you can normalize across the whole market if you prefer)
        scores = segment_df['return'].groupby('instrument').transform(calculate_mad_score).abs().groupby('instrument').max()  # Get absolute max MAD score for each stock in this segment
                
        # Identify stocks that had at least one such event
        black_swan_tickers = scores[scores > MAD_THRESHOLD].sort_values().index.tolist()
                
        # Save Black Swan Dataset
        save_dataset(segment_df, black_swan_tickers[:TOP_K], "extreme", i)

        # Remove Black Swans from pool for remaining categories
        remaining_tickers = [t for t in segment_df.index.get_level_values('instrument').unique() 
                             if t not in black_swan_tickers[:TOP_K]]
        
        if not remaining_tickers:
            continue
        
        ma_segment_df = segment_df.rolling(window=10).mean().dropna()  # Optional: Smooth returns to reduce noise

        # Calculate frequency of positive and negative returns for remaining stocks
        nb_positive_rate = ma_segment_df.loc[(remaining_tickers, slice(None)), 'return'].groupby('instrument').transform(count_positive).groupby('instrument').first()  # Get count of positive returns for each stock
                        
        # Sort tickers
        sorted_tickers = nb_positive_rate.sort_values(ascending=False).index.tolist()
        
        # --- LOGIC 2: UPTREND (RISE) ---
        # Top 300 stocks
        rise_tickers = sorted_tickers[:TOP_K]
        save_dataset(segment_df, rise_tickers, "rise", i)
        
        # --- LOGIC 3: DOWNTREND (FALL) ---
        # Bottom 300 stocks
        fall_tickers = sorted_tickers[-TOP_K:]
        save_dataset(segment_df, fall_tickers, "fall", i)
        
        # --- LOGIC 4: OSCILLATORY (FLUCTUATION) ---
        # Middle 300 stocks
        mid_start = (len(sorted_tickers) // 2) - (TOP_K // 2)
        fluc_tickers = sorted_tickers[mid_start : mid_start + TOP_K]
        save_dataset(segment_df, fluc_tickers, "fluctuation", i)

def save_dataset(full_df, tickers, category, segment_id):
    """
    Saves the specific tickers for this segment as a pickle file.
    Format: dict or DataFrame matching FinTSB expectation.
    """
    if not tickers:
        return
        
    # Extract data for these tickers
    subset_df = full_df.loc[(tickers, slice(None)), :]
    
    # FinTSB expects a dictionary or dataframe object in the pickle.
    # Usually, it's a dict containing 'x' (features) and 'y' (labels), 
    # or simply the dataframe itself if their loader handles it.
    # Based on standard Qlib/FinTSB usage, we save the DataFrame directly.
    
    file_path = os.path.join(TEMP_DIR, category, f"dataset_{segment_id}.pkl")
    
    with open(file_path, "wb") as f:
        pickle.dump(subset_df, f)
        
def calculate_spectral_forecastability(series: pd.Series) -> float:
    """
    Computes the spectral forecastability of a time series using Equation 3 
    from the FinTSB benchmark methodology.
    """
    series = series.dropna()
    N = len(series)
    
    if N < 2:
        raise ValueError("Time series is too short to compute Fourier decomposition.")
        
    # 1. Compute the Fourier decomposition (Power Spectral Density)
    # periodogram returns frequencies from 0 to pi (normalized)
    freqs, power = periodogram(series)
    
    # Drop the DC component (frequency 0) to focus purely on the signal's fluctuations
    power = power[1:]
    
    # 2. Normalize the power spectrum so it sums to 1 (making it a Probability Mass Function)
    power_sum = np.sum(power)
    if power_sum == 0:
        return 1.0  # A perfectly flat line has 0 entropy and maximum predictability
        
    p_discrete = power / power_sum
    
    # 3. Compute continuous differential entropy H(s_i)
    # Because log(2*pi) is the theoretical maximum for a continuous spectrum bounded on [-pi, pi],
    # we must scale the discrete probabilities by the frequency bin width (delta_lambda = 2*pi / N)
    # to accurately approximate the continuous integral H(f) = - integral f(lambda) log(f(lambda)) d_lambda
    delta_lambda = (2 * np.pi) / N
    f_continuous = p_discrete / delta_lambda
    
    # Filter out zero-power bins to avoid log(0) errors
    valid_f = f_continuous[f_continuous > 0]
    valid_p = p_discrete[f_continuous > 0]
    
    # Calculate the scaled entropy H(s_i)
    spectral_entropy = -np.sum(valid_p * np.log(valid_f))
    
    # 4. Compute forecastability phi(s_i) using Equation 3
    phi = 1.0 - (spectral_entropy / np.log(2 * np.pi))
    
    # Constrain bounds (prevent minor floating-point overshoots)
    return float(np.clip(phi, 0.0, 1.0))
        
def calculate_characteristics_for_stock(close_prices):
    """
    Calculates characteristics for a single stock's price sequence.
    """
    if len(close_prices) < 20:
        return {
            "trend_score": 0,
            "autocorr_score": 0,
            "stationarity": 1.0,
            "forecastability": 1.0
        }

    # 1. Trend Strength (Linear Regression R-value)
    x = np.arange(len(close_prices))
    y = close_prices
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    trend_score = abs(r_value) 

    # 2. Autocorrelation (ACF)
    acf_values = acf(close_prices, nlags=10, fft=False)
    autocorr_score = np.sum(np.abs(acf_values[1:]))

    # 3. Stationarity (ADF Test)
    try:
        adf_result = adfuller(close_prices)[0]
    except:
        adf_result = 0 # Default to non-stationary on error
        
    # 4. Forecastability (Spectral Entropy)
    forecastability = calculate_spectral_forecastability(pd.Series(close_prices))

    return {
        "trend_score": trend_score,
        "autocorr_score": autocorr_score,
        "stationarity": adf_result,
        "forecastability": forecastability
    }

def calculate_sequence_characteristics(df):
    """
    Computes the metrics mentioned in FinTSB Section 4.1.1
    Expects df with a 'close' column.
    """
    # We calculate metrics for each stock and then average them to get a segment-level score.
    stocks = df.index.get_level_values("instrument").unique()
    trend_scores = []
    autocorr_scores = []
    stationarity_stats = []
    forecastability_stats = []
    
    for stock in stocks:
        
        stock_df = df.xs(stock, level="instrument")
        close_prices = stock_df["close"].values
        
        if len(close_prices) < 20: # Skip very short sequences
            continue
        
        # Calculate characteristis for this stock
        characteristics = calculate_characteristics_for_stock(close_prices)
        trend_scores.append(characteristics["trend_score"])
        autocorr_scores.append(characteristics["autocorr_score"])
        stationarity_stats.append(characteristics["stationarity"])
        forecastability_stats.append(characteristics["forecastability"])
        

    return {
        "trend_score": np.mean(trend_scores),
        "autocorr_score": np.mean(autocorr_scores),
        "stationarity": np.mean(stationarity_stats),
        "forecastability": np.mean(forecastability_stats)
    }

def score_candidates(regime_name):
    """
    Loads all candidate datasets for a regime, scores them, and picks Top 5.
    """
    regime_dir = os.path.join(TEMP_DIR, regime_name)
    if not os.path.exists(regime_dir):
        print(f"Skipping {regime_name} (folder not found)")
        return

    candidates = []
    
    # Iterate through all generated segments
    files = [f for f in os.listdir(regime_dir) if f.endswith(".pkl")]
    print(f"Scoring {len(files)} candidates for regime: {regime_name}...")

    for f in files:
        file_path = os.path.join(regime_dir, f)
        try:
            with open(file_path, "rb") as pkl:
                df = pickle.load(pkl)
            
            stats = calculate_sequence_characteristics(df)
            stats["filename"] = f
            candidates.append(stats)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # --- SELECTION LOGIC (Section 4.1.1) ---
    # We sort based on what defines the regime.
    
    if regime_name in ["rise", "fall"]:
        # For Trends: We want Strong Trend (High R-val)
        # We can combine them: Score = Trend + Autocorr
        candidates.sort(key=lambda x: x["trend_score"], reverse=True)
        
    elif regime_name == "extreme": # Black Swan
        # For Extreme: We want high volatility or shock. 
        # (The previous script already filtered by >9.5% return).
        # We prioritize high non-stationarity (lowest statistics) 
        candidates.sort(key=lambda x: x["forecastability"], reverse=False)
        
    elif regime_name == "fluctuation": # Oscillatory
        # For Fluctuation: We want LOW Trend (Low R-val) and HIGH Stationarity (Low P-val)
        # So we sort by Trend (Ascending)
        candidates.sort(key=lambda x: x["stationarity"], reverse=True)

    # Select Top 5
    top_5 = candidates[:5]
    return top_5

def save_top_5():
    for regime in ["rise", "fall", "fluctuation", "extreme"]:
        top_5_stats = score_candidates(regime)
        
        if not top_5_stats:
            continue

        # Create output folder
        target_dir = os.path.join(OUTPUT_DIR, regime)
        os.makedirs(target_dir, exist_ok=True)
        
        print(f"--- Top 5 Selected for {regime} ---")
        for i, item in enumerate(top_5_stats):
            original_file = item["filename"]
            print(f"  Rank {i+1}: {original_file} (Trend: {item['trend_score']:.4f}, Stationarity: {item['stationarity']:.4f}, Forecastability: {item['forecastability']:.4f})")
            
            # Copy/Save the file to the "Selected" folder
            src = os.path.join(TEMP_DIR, regime, original_file)
            dst = os.path.join(target_dir, f"dataset_rank{i+1}.pkl")
            
            with open(src, "rb") as f_src:
                data = pickle.load(f_src)
            with open(dst, "wb") as f_dst:
                pickle.dump(data, f_dst)

if __name__ == "__main__":
    print("--- FinTSB Dataset Generator ---")
    initialize_qlib()
    
    # Fetch 15 years of data (modify dates if you have less data)
    data = get_market_data(start_date="2010-01-01", end_date="2023-01-01")
    
    segment_and_save(data)
    save_top_5()
    remove_temp_dir(TEMP_DIR)  # Clean up temporary files
    
    print(f"\nDone! Datasets saved to {OUTPUT_DIR}")
