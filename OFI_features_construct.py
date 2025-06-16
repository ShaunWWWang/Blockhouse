import pandas as pd
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_bid_flow(p_now, p_prev, q_now, q_prev):
    """Computes bid-side order flow.

    :param p_now (float): Bid price at time n.
    :param p_prev (float): Bid price at time n-1.
    :param q_now (float): Bid size at time n.
    :param q_prev (float): Bid size at time n-1.
    :return:
        float: Bid order flows at time n.
    """
    if p_now > p_prev:
        return q_now
    elif p_now == p_prev:
        return q_now - q_prev
    else:  # p_now < p_prev
        return -q_now

def compute_ask_flow(p_now, p_prev, q_now, q_prev):
    """Computes ask-side order flow.

    :param p_now (float): Ask price at time n.
    :param p_prev (float): Ask price at time n-1.
    :param q_now (float): Ask size at time n.
    :param q_prev (float): Ask size at time n-1.
    :return:
        float: Ask order flows at time n.
    """
    if p_now > p_prev:
        return -q_now
    elif p_now == p_prev:
        return q_now - q_prev
    else:  # p_now < p_prev
        return q_now

def compute_ofi(df, max_levels=10, h_seconds=60):
    """Computes multi-level Order Flow Imbalance (OFI) over a fixed time window.

    :param df (pd.DataFrame): The given DataFrame about LOB data.
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.DataFrame: ofi_df with index of ts_event and columns of ofi_level from 1 to 10.
                      The first column is best-level OFI and all others are deeper column OFI without being scaled by average size.

    """
    ofi_levels = [[] for _ in range(max_levels)]

    # Iterate row-by-row to compute changes
    for i in tqdm(range(1, len(df))):
        total_bid_flow = []
        total_ask_flow = []
        # Iterate level-by-level to compute changes
        for lvl in range(max_levels):
            bp_col = f'bid_px_0{lvl}'
            ap_col = f'ask_px_0{lvl}'
            bq_col = f'bid_sz_0{lvl}'
            aq_col = f'ask_sz_0{lvl}'

            p_bid_now = df.loc[i, bp_col]
            p_bid_prev = df.loc[i - 1, bp_col]
            q_bid_now = df.loc[i, bq_col]
            q_bid_prev = df.loc[i - 1, bq_col]

            p_ask_now = df.loc[i, ap_col]
            p_ask_prev = df.loc[i - 1, ap_col]
            q_ask_now = df.loc[i, aq_col]
            q_ask_prev = df.loc[i - 1, aq_col]

            bid_flow = compute_bid_flow(p_bid_now, p_bid_prev, q_bid_now, q_bid_prev)
            ask_flow = compute_ask_flow(p_ask_now, p_ask_prev, q_ask_now, q_ask_prev)
            ofi = bid_flow - ask_flow

            total_bid_flow.append(bid_flow)
            total_ask_flow.append(ask_flow)
            ofi_levels[lvl].append(ofi)

    ofi_df = pd.DataFrame({f'ofi_level_{i + 1}': ofi_levels[i] for i in range(max_levels)}, index = df['ts_event'][1:])
    ofi_df = ofi_df.rolling(f'{h_seconds}s').sum() # Applying a time-based rolling sum to accumulate OFI over the window
    return ofi_df

def compute_average_order_book_depth(df, max_levels = 10, h_seconds = 60):
    """Computes average order book depth over time.

    :param df (pd.DataFrame): The given DataFrame about LOB data.
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.Series: Time-indexed average order book depth.
    """
    total_sum = pd.Series(0, index=df['ts_event'])
    for lvl in range(max_levels):
        bq_col = f'bid_sz_0{lvl}'
        aq_col = f'ask_sz_0{lvl}'
        sz_sum = df[bq_col]+df[aq_col]
        sz_sum.index = df['ts_event']
        sz_sum = sz_sum.rolling(f'{h_seconds}s').mean()/2
        total_sum += sz_sum
    total_sum = total_sum/max_levels
    return total_sum

# These are functions that generate the result using the functions defined above
def get_best_level_OFI(df,max_levels=10, h_seconds=60):
    """Extracts best-level (level 1) OFI.

    :param df (pd.DataFrame): The given DataFrame about LOB data.
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.Series: Best-level OFI time series.
    """
    ofi_df = compute_ofi(df, max_levels=max_levels, h_seconds=h_seconds)
    return ofi_df['ofi_level_1']

def get_scaled_deeper_level_OFI(df,max_levels=10, h_seconds=60):
    """Computes scaled deeper-level OFI by dividing by average order book depth.

    :param df (pd.DataFrame): The given DataFrame about LOB data.
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.Series: Scaled OFI for each level.
    """
    ofi_df = compute_ofi(df, max_levels=max_levels, h_seconds=h_seconds)
    aobd = compute_average_order_book_depth(df, max_levels=max_levels, h_seconds=h_seconds)
    aobd = aobd[1:]
    scaled_ofi = ofi_df.div(aobd, axis=0)
    return scaled_ofi


def get_integrated_OFI(df, max_levels=10, h_seconds=60):
    """Computes integrated OFI using PCA on scaled OFI.

    :param df (pd.DataFrame): The given DataFrame about LOB data.
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.Series: Integrated OFI time series.
    """
    scaled_OFI = get_scaled_deeper_level_OFI(df, max_levels=max_levels, h_seconds=h_seconds)
    scaler = StandardScaler()
    standardized_ofi = scaler.fit_transform(scaled_OFI)
    pca = PCA(n_components=1)
    ofi_pca = pca.fit_transform(standardized_ofi)
    # Get the first principal vector (weights)
    w1 = pca.components_[0]  # Shape: (n_levels,)
    w1_normalized = w1 / sum(abs(w1))  # Ensure weights sum to 1
    # Multiply scaled OFI by weights and sum across levels
    integrated_ofi = pd.Series(
        standardized_ofi @ w1_normalized,  # Matrix multiplication
        index=scaled_OFI.index,
        name="integrated_ofi"
    )
    return integrated_ofi

def get_cross_asset_OFI(df, type='best', max_levels=10, h_seconds=60):
    """Computes cross-asset OFI for all symbols in the dataset.

    :param df (pd.DataFrame): DataFrame with LOB data including a 'symbol' column.
    :param type (str): Type of OFI to compute ('best' or 'integrated').
    :param max_levels (int): The deepest price level according to the data (default 10).
    :param h_seconds (int): A given time interval (in seconds) used to calculates the accumulative OFIs (default 60).
    :return:
        pd.DataFrame: A DataFrame where each column is an asset's OFI series, aligned by timestamp.
    """
    grouped = df.groupby(['symbol'])
    dict = {}
    for asset, group in grouped:
        if type == 'best':
            result = get_best_level_OFI(group, max_levels=max_levels, h_seconds=h_seconds)
        elif type == 'integrated':
            result = get_integrated_OFI(group, max_levels=max_levels, h_seconds=h_seconds)
        else:
            raise ValueError("type must be either 'best' or 'integrated'")
        dict[asset] = result
    cross_asset_ofi = pd.DataFrame.from_dict(dict, orient='columns')
    return cross_asset_ofi

def main():
    """Main execution function for computing OFI features from LOB data."""
    df = pd.read_csv('first_25000_rows.csv')
    df = df.sort_values(by='ts_event').reset_index(drop=True)
    df['ts_event'] = pd.to_datetime(df['ts_event'], unit='ns')

    best_level_OFI = get_best_level_OFI(df)
    scaled_OFI = get_scaled_deeper_level_OFI(df)
    integrated_OFI = get_integrated_OFI(df)
    cross_asset_OFI = get_cross_asset_OFI(df)
    print("Best-level OFI:\n", best_level_OFI.head())
    print("\nScaled Deeper-level OFI:\n", scaled_OFI.head())
    print("\nIntegrated OFI:\n", integrated_OFI.head())
    print("\nCross-Asset OFI:\n", cross_asset_OFI.head())

if __name__ == "__main__":
    main()