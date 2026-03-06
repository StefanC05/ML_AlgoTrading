# feature_lib.py
# =============================================================================
# LIBRARY: FEATURE ENGINEERING & UTILS
# STATUS: FINAL (Refactored to modular functions + statsmodels frac_diff)
# =============================================================================

import numpy as np
import pandas as pd
import talib
import torch
from sklearn.mixture import GaussianMixture

# Import statsmodels for fractional differencing
from statsmodels.tsa.statespace.tools import diff

# Using manual TBM implementation (no external dependencies)


def manual_get_events(
    close: pd.Series,
    t_events,
    pt_sl,
    target,
    min_ret: float,
    vertical_barrier_times=None,
) -> pd.DataFrame:
    """Simplified manual TBM Events (Marco de Prado "Machien learning...something something" you know it!)."""
    events = pd.DataFrame(index=t_events)
    events["t1"] = vertical_barrier_times.reindex(t_events).fillna(close.index[-1])
    events["trgt"] = target.reindex(t_events)
    events = events[events["trgt"] > min_ret]

    return events


def manual_get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    Calculate log returns and binary labels for triple barrier method events.

    For each event, computes the return from entry to exit and creates binary labels.
    Handles NaN values and ensures index integrity for vectorized operations.
    """
    # 1: Clean events - missing t1 (exit time)
    events_clean = events.dropna(subset=["t1"])

    # 2: Get prices at entry (events.index) and exit (t1) points
    entry_prices = close.loc[events_clean.index]  # Price when event started
    exit_prices = close.loc[events_clean["t1"]]  # Price when event ended

    # 3: Calculate log returns
    log_returns = np.log(exit_prices.values) - np.log(entry_prices.values)

    # output DataFrame with results
    results = pd.DataFrame(index=events_clean.index)
    results["ret"] = log_returns  # Actual returns
    results["bin"] = np.sign(log_returns)  # Binary labels: +1 (profit), -1 (loss)

    return results


def ticker_count(obj) -> int:
    """Count unique elements

    Handles df (counts unique tickers), GroupBy (counts groups),
    and Series (counts total elements).
    """  # ...what a mess..are you sure? are you absolut sure? realy?
    if isinstance(obj, pd.DataFrame):
        # MultiIndex DataFrame: count unique tickers in first level
        return obj.index.get_level_values("ticker").nunique()
    elif hasattr(obj, "ngroups"):
        # GroupBy object: count number of groups
        return obj.ngroups
    elif isinstance(obj, pd.Series):
        # Series: count total number of elements
        return len(obj)
    else:
        raise TypeError(
            "Unsupported object type. Expected DataFrame, GroupBy, or Series."
        )


def cleanup_features(features: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with excessive NaN values from features DataFrame."""
    print(f"Processing {ticker_count(features)} tickers")
    print(features.sample(25))  # Show random sample of data

    before_count = len(features)
    # Require at least 50% of columns to be non-NaN
    thresh = int(0.5 * len(features.columns))
    features.dropna(thresh=thresh, inplace=True)

    removed_count = before_count - len(features)
    print(f"Removed {removed_count} rows with < {thresh} non-NaN values")
    return features


# --- FEATURE GROUPS ---


def add_basic_features(
    prices: pd.DataFrame, features: pd.DataFrame, intervals: list[int]
) -> pd.DataFrame:
    # asic price-based and temporal features
    print(f"Calculating basic features for {ticker_count(prices)} tickers")

    # Ensure data is sorted for time-based calculations
    prices = prices.sort_index()

    # Group by ticker for per-stock calculations
    close_grouped = prices.groupby("ticker")["close"]

    # Price-based features for each time interval
    for t in intervals:
        # Simple percentage returns
        features[f"ret_{t}"] = close_grouped.pct_change(t)
        # Log returns (more suitable for financial modeling)
        features[f"log_ret_{t}"] = np.log(
            close_grouped.shift(0) / close_grouped.shift(t)
        )
        # Cross-sectional rank percentile (relative strength)
        features[f"ret_rel_perc_{t}"] = (
            features[f"ret_{t}"].groupby("date").rank(pct=True) * 20
        )

    # Temporal features from date index
    dates = prices.index.get_level_values("date")
    features["month"] = dates.month
    features["day"] = dates.day
    # Cyclic weekday encoding using sine/cosine (handles weekend discontinuity)
    features["weekday_sin"] = np.sin(2 * np.pi * (dates.weekday % 5) / 5)
    features["weekday_cos"] = np.cos(2 * np.pi * (dates.weekday % 5) / 5)

    print(f"Basic features completed for {ticker_count(features)} tickers")
    return features


def add_complex_features(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    intervals: list[int],
    gmm_components: int = 3,
    skip_gmm: bool = False,
) -> pd.DataFrame:
    """Calculate complex technical and statistical features for each ticker."""
    print(f"Calculating complex features for {ticker_count(prices)} tickers")

    # 1. TECHNICAL INDICATORS
    def _calc_tech(df) -> pd.DataFrame:
        """technical indicators for ticker."""

        ti_df = pd.DataFrame(index=df.index)
        c, h, l = df["close"], df["high"], df["low"]

        ti_df["RSI"] = talib.RSI(c, 14)
        ti_df["NATR"] = talib.NATR(h, l, c, 14)  # Normalized Average True Range
        upper, _, lower = talib.BBANDS(c, 20)
        ti_df["bbu"] = upper / c  # Upper band ratio
        ti_df["bbl"] = c / lower  # Lower band ratio (inverted)

        return ti_df

    # Apply technical indicators per ticker and merge
    tech_features = prices.groupby("ticker", group_keys=False).apply(_calc_tech)
    features = pd.concat([features, tech_features], axis=1)

    # Advanced Features - IMPROVED
    def _calc_adv(df: pd.DataFrame) -> pd.DataFrame:
        """Congratulations..Its a wrapper.
        1. IBS - CPrice position within daily range(0(low) - 1(hgh)) not sooo advanced
         2. Parkinson Volatility - Volatility based on high-low prices
         3. VAM (Volatility Adjusted Momentum) - Momentum adjusted by volatility
         4. Variance Ratio - Test for random walk hypothesis
         5. Momentum Ratio - ip down day ratio

         df :   all ticker prices holc+v

         Returns:   result:df
        """

        result = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]

        result["ibs"] = _calc_ibs(close, high, low)

        # Parkinson volatility is a more robust volatility, based on high-low prices
        # VAM (Volatility Adjusted Momentum)
        print("Calculating Parkinson Volatility and VAM...")
        volatility_features = _calc_volatility_features(close, high, low, intervals)
        result = pd.concat([result, volatility_features], axis=1)

        # Momentum Ratio --- positive to negative returns over different time periods
        print(" Momentum Ratio...")
        momentum_features = _calc_momentum_ratio(close, intervals)
        result = pd.concat([result, momentum_features], axis=1)

        # Variance Ratio # Tests the random walk hypothesis by comparing variances over different periods
        # Only calculated for intervals > 1 to avoid division by zero
        # print("      Calculating Variance Ratio...")
        variance_features = _calc_variance_ratio(close, intervals)
        result = pd.concat([result, variance_features], axis=1)

        return result

    def _calc_ibs(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
        """
        IBS measures the position of the closing price within the day's trading range.
        It's a normalized value between 0 and 1.

        Returns:pd.Series        IBS values, with NaN where high equals low
        """
        try:
            # Calculate range and avoid division by zero
            price_range = high - low
            ibs = (close - low) / price_range

            # Replace division by zero with NaN
            ibs = ibs.replace([np.inf, -np.inf], np.nan)

            # Ensure values are within [0, 1] range (handle small floating point errors)
            ibs = np.clip(ibs, 0, 1)

            return ibs

        except Exception as e:
            print(f"Warning: Error calculating IBS: {e}")
            return pd.Series([np.nan] * len(close), index=close.index)

    def _calc_volatility_features(
        close: pd.Series, high: pd.Series, low: pd.Series, intervals: list
    ) -> pd.DataFrame:
        """
        Parkinson volatility and VAM (Volatility Adjusted Momentum).
                Parkinson: high-low prices to estimate volatility.
                VAM adjusts momentum by dividing by volatility (ATR).

        beeing honest: i have no clue to calculate it befor i copie it from the all knowing and magic net....
        i dont need to know it? i just fill the pipline......

          Returns: pd.df            df with Parkinson volatility and VAM features
        """
        volatility_df = pd.DataFrame(index=close.index)

        try:
            # Based on the relationship between log returns and price ranges
            const = 1.0 / (4.0 * np.log(2.0))
            log_range_squared = np.log(high / low) ** 2

            # ATR for volatility adjustment
            atr = talib.ATR(high, low, close, 14)

            # Calculate Parkinson volatility for each interval
            for k in intervals:
                # Parkinson volatility formula sqrt(const * mean(log(high/low)^2))
                parkinson_vol = np.sqrt(const * log_range_squared.rolling(k).mean())
                volatility_df[f"parkinson_{k}"] = parkinson_vol

            # Calculate VAM
            for k in intervals:
                momentum = close.pct_change(k)
                vam = momentum / atr

                # Handle division by zero and extreme values
                vam = vam.replace([np.inf, -np.inf], np.nan)
                vam = np.clip(vam, -10, 10)  # Cap extreme values

                volatility_df[f"vam_{k}"] = vam

        except Exception as e:
            print(f" Warning: Error calculating volatility features: {e}")
            # Return empty DataFrame with correct columns
            for k in intervals:
                volatility_df[f"parkinson_{k}"] = np.nan
                volatility_df[f"vam_{k}"] = np.nan

        return volatility_df

    def _calc_momentum_ratio(close: pd.Series, intervals: list) -> pd.DataFrame:
        """
         "momentum ratio" - ratio of positive to negative returns.

        This measures the "balance" between up and down days over X periods.
        simple summring,not a big deal
        Adding 1 to both numerator and denominator avoids division by zero.

        Returns:  momentum_df   with momentum ratio features
        """
        momentum_df = pd.DataFrame(index=close.index)

        try:
            daily_returns = close.pct_change()

            for k in intervals:
                # Count positive and negative returns over the interval
                positive_days = (daily_returns > 0).rolling(k).sum()
                negative_days = (daily_returns < 0).rolling(k).sum()
                # edge cases
                momentum_ratio = (positive_days + 1) / (negative_days + 1)
                momentum_ratio = momentum_ratio.replace([np.inf, -np.inf], np.nan)

                momentum_df[f"mom_ratio_{k}"] = momentum_ratio

        except Exception as e:
            print(f"Warning: Error calculating momentum ratio: {e}")
            # Return empty DataFrame with correct columns
            for k in intervals:
                momentum_df[f"mom_ratio_{k}"] = np.nan

        return momentum_df

    def _calc_variance_ratio(close: pd.Series, intervals: list) -> pd.DataFrame:
        """
         " variance ratio compares the variance of k-period returns
        to k times the variance of 1-period returns.
        A ratio of 1 suggests random walk behavior."

        close :          Closing prices
        intervals : list of time intervals (only k > 1) 3,5,10

        Return: variance_df only the variances
        """
        variance_df = pd.DataFrame(index=close.index)

        try:
            # Calculate 1-day log returns for baseline variance
            log_ret_1d = np.log(close / close.shift(1))

            for k in intervals:
                if k <= 1:
                    continue  # Skip k=1 as it would result in division by zero

                # Calculate k-period log returns
                log_ret_k = np.log(close / close.shift(k))
                var_k = log_ret_k.rolling(k).var()
                var_1 = log_ret_1d.rolling(k).var()

                # Calculate variance ratio
                # For random walk, variance should scale linearly with time
                var_ratio = var_k / (k * var_1)

                # Handle edge cases
                var_ratio = var_ratio.replace([np.inf, -np.inf], np.nan)
                var_ratio = np.clip(
                    var_ratio, 0, 5
                )  # Reasonable bounds for variance ratio

                variance_df[f"var_ratio_{k}"] = var_ratio

        except Exception as e:
            print(f"Warning: Error  variance ratio: {e}")
            # Return empty DataFrame with correct columns for k > 1
            for k in intervals:
                if k > 1:
                    variance_df[f"var_ratio_{k}"] = np.nan

        return variance_df

    # 2. ADVANCED STATISTICAL FEATURES (applied per ticker)
    advanced_features = prices.groupby("ticker", group_keys=False).apply(_calc_adv)
    features = pd.concat([features, advanced_features], axis=1)

    # 3. FRACTIONAL DIFFERENCING (stationarity improvement)
    print("Calculating fractional differencing...")

    def _fractional_diff(x):
        """Apply fractional differencing with d=0.4 to make series more stationary."""
        return pd.Series(diff(x, k_diff=0.4), index=x.index)

    # Apply fractional differencing per ticker, then calculate returns
    frac_diff_prices = prices.groupby("ticker", group_keys=False)["close"].apply(
        _fractional_diff
    )
    frac_grouped = frac_diff_prices.groupby("ticker")

    for t in intervals:
        # Calculate percentage changes on fractionally differenced prices
        features[f"frac_ret_{t}"] = frac_grouped.pct_change(t)

    # 4. MARKET REGIME CLASSIFICATION (unsupervised clustering)
    if not skip_gmm:  # Skip GMM when called globally to prevent future leakage
        if "NATR" in features.columns and "RSI" in features.columns:
            # Use volatility (NATR) and momentum (RSI) for regime detection
            regime_data = features[["NATR", "RSI"]].dropna()
            if not regime_data.empty:
                # Fit Gaussian Mixture Model to identify market regimes
                gmm = GaussianMixture(n_components=gmm_components, random_state=42)
                gmm.fit(regime_data)
                # Assign regime labels (0, 1, 2, etc.) to each observation
                features.loc[regime_data.index, "market_regime"] = gmm.predict(
                    regime_data
                )

    print(f"Complex features completed for {ticker_count(features)} tickers")
    return features


def add_complex_features_skip_gmm(
    prices: pd.DataFrame, features: pd.DataFrame, intervals: list, gmm_components=3
) -> pd.DataFrame:
    """Wrapper for add_complex_features that skips GMM fitting to prevent future leakage."""
    return add_complex_features(
        prices, features, intervals, gmm_components=gmm_components, skip_gmm=True
    )


def create_global_features(
    prices: pd.DataFrame, intervals: list, gmm_components: int = 3
) -> pd.DataFrame:
    """
    Create all features that are safe for global application (no future leakage).
    Apply this to full dataset (2000-2025).
    """
    features = pd.DataFrame(index=prices.index)

    # All safe operations
    features = add_basic_features(prices, features, intervals)
    features = add_complex_features_skip_gmm(
        prices, features, intervals, gmm_components=gmm_components
    )
    features = add_targets(prices, features, intervals)
    features = cleanup_features(features)

    return features


def create_per_fold_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame, gmm_components: int = 3
) -> tuple:
    """non globalise features. futere leacking etc....
    Create features that require fitting only on training data (per-fold).
    Call this inside each fold after winsorizing.

            X_train : Training features
            X_val : Validation features
            gmm_components : Number of GMM components

    Returns:     X_train, X_val:
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    # Market regime classification (GMM) - fit only on training data
    if "NATR" in X_train.columns and "RSI" in X_train.columns:
        regime_data_train = X_train[["NATR", "RSI"]].dropna()

        if not regime_data_train.empty and len(regime_data_train) > 100:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=gmm_components, random_state=42)
            gmm.fit(regime_data_train)

            # Apply to training data
            X_train.loc[regime_data_train.index, "market_regime"] = gmm.predict(
                regime_data_train
            )

            # Apply same model to validation data
            regime_data_val = X_val[["NATR", "RSI"]].dropna()
            if not regime_data_val.empty:
                X_val.loc[regime_data_val.index, "market_regime"] = gmm.predict(
                    regime_data_val
                )

    return X_train, X_val


def add_targets(
    prices: pd.DataFrame, features: pd.DataFrame, intervals: list
) -> pd.DataFrame:
    """Create prediction targets for machine learning models."""
    print(f"Creating targets for {ticker_count(prices)} tickers")

    # Group close prices by ticker for target calculations
    close_grouped = prices.groupby("ticker")["close"]

    # 1. STANDARD REGRESSION TARGETS (forward-looking returns)
    for t in intervals:
        # Forward percentage returns
        features[f"fwd_ret_{t:02}"] = close_grouped.pct_change(t).shift(-t)
        # fwd log returns
        features[f"fwd_log_ret_{t:02}"] = np.log(
            close_grouped.shift(-t) / close_grouped.shift(0)
        )

        # Forward volatility ratios
        if t > 1:
            current_vol = (
                close_grouped.pct_change().rolling(20).std()
            )  # Recent volatility

            future_vol = (
                close_grouped.pct_change().rolling(t).std().shift(-t)
            )  # Future volatility

            features[f"fwd_vola_{t:02}"] = future_vol / current_vol  # Volatility ratio

    # 2. TRIPLE BARRIER METHOD TARGETS (classification with barriers)
    try:
        print("Generating Triple Barrier Method labels...")
        # Calculate rolling volatility for dynamic barriers
        volatility = close_grouped.pct_change().rolling(50).std()

        # Process each ticker individually
        for ticker, ticker_data in prices.groupby(level="ticker"):
            close_prices = ticker_data["close"]
            ticker_vol = volatility.loc[ticker].shift(1)  # Use lagged volatility
            event_times = close_prices.index

            for t in intervals:
                # Calculate time barrier (t days ahead)
                time_barrier_idx = close_prices.index.searchsorted(
                    close_prices.index + pd.Timedelta(days=t)
                )
                time_barrier_idx = time_barrier_idx[
                    time_barrier_idx < len(close_prices.index)
                ]
                time_barriers = pd.Series(
                    close_prices.index[time_barrier_idx],
                    index=close_prices.index[: len(time_barrier_idx)],
                )

                # Generate triple barrier events using manual implementation
                events = manual_get_events(
                    close_prices,
                    event_times,
                    pt_sl=[3.5, 2.0],
                    target=ticker_vol,
                    min_ret=0.005,
                    vertical_barrier_times=time_barriers,
                )
                labels = manual_get_bins(events, close_prices)

                # Store TBM labels and returns
                features.loc[labels.index, f"tbm_label_{t:02}"] = labels[
                    "bin"
                ]  # Binary labels

                features.loc[labels.index, f"tbm_ret_{t:02}"] = labels[
                    "ret"
                ]  # Actual returns..why again?

    except Exception as e:
        print(f"Skipping TBM targets, due to error: {e}")

    for t in intervals:
        target_col = f"fwd_log_ret_{t:02}"
        if target_col in features.columns:
            # smoothed  returns to labels. using tanh activation bounded outputs [-1,1]
            raw_values = features[target_col].fillna(0).values
            tensor_vals = torch.tensor(raw_values, dtype=torch.float32)
            dl_labels = torch.tanh(tensor_vals * 100.0).numpy()  # Scale by 100

            features[f"dl_label_{t:02}"] = dl_labels

    print(f"Creation for {ticker_count(features)} tickers complete")
    return features
