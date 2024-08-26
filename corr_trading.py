# Standard libraries
import datetime
import os
import re
import time
import random
import requests
from itertools import product, groupby, combinations
from datetime import datetime, date

# Data manipulation libraries
import numpy as np
import pandas as pd

# Statistical and time series analysis libraries
import scipy.stats as stats
from statsmodels.api import OLS
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from scipy.stats import norm

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
import sqlite3

############################################################################################################################################################

def load_sql_table (table, path) :
    conn = sqlite3.connect(path)

    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    
    date_columns = [col for col in df.columns if 'DATE' in col.upper()]

    for col in date_columns:
        df[col] = pd.to_datetime(df[col],errors='coerce')
    
    conn.close()
    
    return df

############################################################################################################################################################


def read_fin_data(asset_classes, path):

    fin_data = pd.DataFrame()
    
    sql_fin_data_dict = {'equity':'EQUITY_PRICES',
                         'index':'INDICES',
                         'funds':'FUNDS',
                         'fiat':'CURRENCY_FIAT',
                         'crypto':'CURRENCY_CRYPTO'}
    
    if asset_classes != 'all':
        for asset in asset_classes:
            temp = load_sql_table(sql_fin_data_dict[asset], path)
            temp['asset_class'] = asset
            fin_data = pd.concat([temp,fin_data])
            
            print(f"{asset} asset class read")
            
    elif asset_classes != 'all':
        for asset in list(sql_fin_data_dict.keys()):
            temp = load_sql_table(sql_fin_data_dict[asset])
            temp['asset_class'] = asset
            fin_data = pd.concat([temp,fin_data])
            
    return fin_data
            
# def read_fin_data(directory, asset_classes):
#     """
#     List all CSV and XLSX files in the specified directory and its subdirectories,
#     optionally filtering by asset class to find and read the latest financial data file.

#     Parameters:
#     - directory (str): Path to the directory containing CSV and XLSX files.
#     - asset_class (str): The asset class to filter files by. Options are 'all', 'equity', or any other specific asset class.
#       - 'all': Combine all relevant files and read the most recent one.
#       - 'equity': Specifically look for files related to equity with 'EQUITY' and 'PRICES' in the filename.
#       - Any other asset class: Filter files based on the asset class name in the filename.

#     Returns:
#     - tuple[str, pd.DataFrame]:
#       - str: Path to the latest financial data file based on the asset class filter.
#       - pd.DataFrame: DataFrame containing the financial data read from the latest file.
#     """

#     file_list = []

#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             if file.endswith(".csv") or file.endswith(".xlsx"):
#                 file_list.append(os.path.join(root, file))

#     fin_data = pd.DataFrame()

#     if asset_classes == "all":
        
#         # Contains the date string of the latest extract
#         temp_date_str = sorted(file_list)[-1][-12:-4]

#         latest_fin_data_file = sorted(
#             file
#             for file in file_list
#             if all(exclusion not in file for exclusion in ["CF", "IS", "BS"])
#             and temp_date_str in file
#         )
#         temp_asset_class = [
#             path.split("/")[-1].split()[0].lower() for path in latest_fin_data_file
#         ]


#         for x, y in zip(latest_fin_data_file, temp_asset_class):
#             temp = pd.read_csv(x)
#             temp["asset_class"] = y
#             fin_data = pd.concat([fin_data, temp])

#     if asset_classes != "all":

#         for asset in asset_classes:
            
#             latest_fin_data_file = sorted(
#                 [file for file in file_list if asset.upper() in file])[-1]

#             if asset == "equity":
#                 latest_fin_data_file = sorted(
#                     [file for file in file_list if "EQUITY" in file and "PRICES" in file])[-1]

#             temp = pd.read_csv(latest_fin_data_file)
#             temp['asset_class'] = asset
#             fin_data = pd.concat([temp,fin_data])
            
#             print(f"{asset} asset class read")

#     fin_data.drop_duplicates(subset=["Ticker", "Date"], inplace=True)
#     fin_data["Date"] = pd.to_datetime(fin_data["Date"])

#     return fin_data


############################################################################################################################################################


def remove_dropped(df, dropped_df):

    df_filtered = df[~df["Ticker"].isin(dropped_df["Ticker"])]
    print("Dropped tickers removed")

    return df_filtered


############################################################################################################################################################


def filter_years(df, all_tickers_df, years_active, start_time):

    # Filter to tickers that have been public for the the last x years

    temp_date = start_time - relativedelta(months=12 * years_active)

    temp = all_tickers_df[all_tickers_df["firstTradeDateEpochUtc"] <= temp_date]
    
    df_filtered = df[df["Ticker"].isin(temp["Ticker"])]
    print(f"Filtered to years_active > {years_active}")

    return df_filtered


############################################################################################################################################################


def filter_most_traded(df, top_n=100):
    max_date = df["Date"].max()
    top_equity = df[(df["Date"] == max_date) & (df["asset_class"] == "equity")]
    top_equity = top_equity.sort_values(by="Volume", ascending=False).head(top_n)

    df2 = pd.concat(
        [df[df["asset_class"] != "equity"], df[df["Ticker"].isin(top_equity["Ticker"])]]
    )
    print(f"Equity stocks filtered to Top {top_n} most traded")
    return df2


############################################################################################################################################################


def handle_nan(df):
    """
    Interpolate NaN values in the DataFrame 'df' using linear interpolation method.

    Parameters:
    - df (DataFrame): Input DataFrame containing numerical data with NaN values.

    Returns:
    - DataFrame: DataFrame with NaN values interpolated using linear interpolation method
                 between the nearest valid data points.
    """
    # Copy the DataFrame to avoid modifying the original
    df = df.copy()

    # Identify NaN values in the middle of the data
    nan_indices = df.isna().all(axis=1)

    # Interpolate NaN values with the mean between the value before and after it
    for col in df.columns:
        df[col] = df[col].interpolate(method="linear", limit_area="inside")

    print(f"Missing values interpolated.")

    return df


############################################################################################################################################################
def new_combinations_list(a, b):
    b_set = set(
        tuple(sublist) for sublist in b
    )  # Convert each sublist in b to a tuple and add to a set
    result = [
        sublist
        for sublist in a
        if tuple(sublist) not in b_set and len(set(sublist)) == len(sublist)
    ]
    return result


############################################################################################################################################################
# Function to perform Engle-Granger cointegration test
def engle_granger_cointegration(ts1, ts2):
    result = coint(ts1, ts2)
    return result[1], result[0]  # return p-value and cointegration score


############################################################################################################################################################
# Function to update Engle-Granger cointegration results
def calculate_engle_granger(
    recalculate_engle_granger,
    ticker_list,
    fin_data3,
    engle_granger_sig_level,
    project_path,
):

    engle_granger_file = sorted(
        os.listdir(f"{project_path}/Outputs/1_Pairing/Cointegration Test/Engle Granger")
    )
    engle_granger_file = [
        file for file in engle_granger_file if file.startswith(f"Engle Granger Test")
    ][-1]
    engle_granger_results = pd.read_csv(
        f"{project_path}/Outputs/1_Pairing/Cointegration Test/Engle Granger/{engle_granger_file}"
    )

    all_combinations = list(combinations(ticker_list, 2))
    all_combinations = [list(pair) for pair in all_combinations if pair[0] != pair[1]]

    if recalculate_engle_granger == "ALL":
        print(f"{len(all_combinations)} new combinations to be calculated")

        # Will recalculate all pairs that are in ticker_list. WILL NOT DELETE ALL RESULTS OF PREVIOUS TESTS.

        # List to store results
        temp = []

        # Calculate cointegration for all pairs
        for i, j in all_combinations:

            stock1 = i
            stock2 = j
            ts1 = fin_data3[stock1]
            ts2 = fin_data3[stock2]
            start_date = min(ts1.index[0], ts2.index[0])
            end_date = max(ts1.index[-1], ts2.index[-1])
            pvalue, score = engle_granger_cointegration(
                ts1, ts2
            )  # This function is defined within this script
            temp.append(
                {
                    "stock1": stock1,
                    "stock2": stock2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "score": score,
                    "p-value": pvalue,
                    "cointegrated": pvalue
                    < engle_granger_sig_level,  # Assuming 0.05 as the threshold for cointegration
                }
            )

        # Convert results to DataFrame
        temp = pd.DataFrame(temp)
        temp["date_calculated"] = datetime.now().strftime("%Y-%m-%d")
        temp = temp[
            [
                "stock1",
                "stock2",
                "start_date",
                "end_date",
                "date_calculated",
                "score",
                "p-value",
                "cointegrated",
            ]
        ]

        engle_granger_results = pd.concat([engle_granger_results, temp])

        engle_granger_results.drop_duplicates(
            subset=["stock1", "stock2"], keep="last", inplace=True
        )

        engle_granger_results.to_csv(
            f'{project_path}/Outputs/1_Pairing/Cointegration Test/Engle Granger/Engle Granger Test {datetime.now().strftime("%Y-%m-%d")}.csv',
            index=False,
        )

        print("\033[1mCointegration results calculated for pairs in ticker_list\033[0m")

    elif recalculate_engle_granger == "NEW":

        # List of pairs that have been calculated prior
        calculated_combinations = []
        calculated_combinations = engle_granger_results[
            ["stock1", "stock2"]
        ].values.tolist()

        new_combinations = new_combinations_list(
            all_combinations, calculated_combinations
        )
        print(f"{len(new_combinations)} new combinations to be calculated")

        # Initialize an empty list to store results
        temp = []

        # Calculate cointegration for all pairs
        for i, j in new_combinations:

            stock1 = i
            stock2 = j

            ts1 = fin_data3[stock1]
            ts2 = fin_data3[stock2]
            start_date = min(ts1.index[0], ts2.index[0])
            end_date = max(ts1.index[-1], ts2.index[-1])
            pvalue, score = engle_granger_cointegration(ts1, ts2)
            temp.append(
                {
                    "stock1": stock1,
                    "stock2": stock2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "score": score,
                    "p-value": pvalue,
                    "cointegrated": pvalue
                    < engle_granger_sig_level,  # Assuming 0.05 as the threshold for cointegration
                }
            )

        if len(temp) != 0:
            # Convert results to DataFrame
            temp = pd.DataFrame(temp)
            temp["date_calculated"] = datetime.now().strftime("%Y-%m-%d")
            temp = temp[
                [
                    "stock1",
                    "stock2",
                    "start_date",
                    "end_date",
                    "date_calculated",
                    "score",
                    "p-value",
                    "cointegrated",
                ]
            ]

            engle_granger_results = pd.concat([engle_granger_results, temp])

            engle_granger_results.drop_duplicates(
                subset=["stock1", "stock2"], keep="last", inplace=True
            )

            for col in engle_granger_results.columns:
                if "date" in col.lower():
                    engle_granger_results[col] = pd.to_datetime(
                        engle_granger_results[col], format="mixed"
                    )

            engle_granger_results.to_csv(
                f'{project_path}/Outputs/1_Pairing/Cointegration Test/Engle Granger/Engle Granger Test {datetime.now().strftime("%Y-%m-%d")}.csv',
                index=False,
            )
        else:
            print("\033[1mCointegration results exist for all combinations\033[0m")

    elif recalculate_engle_granger == "NO":

        print("\033[1mCointegration results loaded\033[0m")

    for col in engle_granger_results.columns:
        if "date" in col.lower():
            engle_granger_results[col] = pd.to_datetime(
                engle_granger_results[col], format="mixed"
            )
    return engle_granger_results


############################################################################################################################################################


# Perform Johansen cointegration test for a pair of assets
def johansen_cointegration_test(pair_data, det_order, k_ar_diff, significance_level):
    """
    Perform Johansen cointegration test for a pair of assets.

    Parameters:
    pair_data: pandas DataFrame, shape (n_obs, 2)
        The input data where each column represents a time series of prices or returns for one asset pair.
    det_order: int
        Deterministic trend order to include in the model. Typically 0 for no deterministic trend,
        1 for a constant term, 2 for both constant and linear trend.
    k_ar_diff: int
        Lag order to include in the model.
    significance_level: float
        Significance level to use for critical values. For example, 0.05 for a 95% confidence level.

    Returns:
    result: dict
        Dictionary containing maximum eigenvalue, trace statistic, critical values for max_eigen and trace statistic.
    """
    result = {}

    # Perform Johansen test
    johansen_results = coint_johansen(pair_data, det_order, k_ar_diff)

    # Get max eigenvalue statistic
    result["max_eigen"] = johansen_results.lr1[0]

    # Get trace statistic
    result["trace_statistic"] = johansen_results.lr2[0]

    # Determine the index for the significance level
    sig_level_index = {0.90: 0, 0.95: 1, 0.99: 2}.get(
        1 - significance_level, 1
    )  # Default to 95% if invalid level

    # Get critical values
    critical_values = johansen_results.cvt
    result["critical_value_max_eigen"] = critical_values[0, sig_level_index]
    result["critical_value_trace_statistic"] = critical_values[1, sig_level_index]

    return result


############################################################################################################################################################
def calculate_johansen(
    recalculate_johansen,
    ticker_list,
    fin_data3,
    project_path,
    det_order=0,
    k_ar_diff=1,
    johansen_sig_level=0.05,
):

    global asset_pair
    
    johansen_file = sorted(
        os.listdir(f"{project_path}/Outputs/1_Pairing/Cointegration Test/Johansen")
    )
    johansen_file = [
        file for file in johansen_file if file.startswith(f"Johansen Test")
    ][-1]
    johansen_results = pd.read_csv(
        f"{project_path}/Outputs/1_Pairing/Cointegration Test/Johansen/{johansen_file}"
    )

    all_combinations = list(combinations(ticker_list, 2))
    all_combinations = [list(pair) for pair in all_combinations if pair[0] != pair[1]]

    if recalculate_johansen == "ALL":

        print(f"{len(all_combinations)} new combinations to be calculated")

        # Set Johansen test parameters
        # det_order No deterministic trend
        # k_ar_diff # Lag order
        # johansen_sig_level

        # Initialize an empty list to store results
        temp = []

        # Perform Johansen cointegration test for all pairs of assets
        for i, j in all_combinations:
            stock1 = i
            stock2 = j
            # Select the pair of assets to test
            asset_pair = fin_data3[[stock1, stock2]]
            start_date = min(asset_pair.index[0], asset_pair.index[0])
            end_date = max(asset_pair.index[-1], asset_pair.index[-1])

            # Perform Johansen cointegration test
            result = johansen_cointegration_test(
                asset_pair.values, det_order, k_ar_diff, johansen_sig_level
            )

            # Determine if cointegrated
            cointegrated = (
                result["max_eigen"] > result["critical_value_max_eigen"]
            ) and (result["trace_statistic"] > result["critical_value_trace_statistic"])

            # Append results to list
            temp.append(
                {
                    "stock1": stock1,
                    "stock2": stock2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "max_eigen": result["max_eigen"],
                    "trace_statistic": result["trace_statistic"],
                    "critical_value_max_eigen": result["critical_value_max_eigen"],
                    "critical_value_trace_statistic": result[
                        "critical_value_trace_statistic"
                    ],
                    "cointegrated": cointegrated,
                }
            )

        # Create DataFrame from results
        temp = pd.DataFrame(temp)

        temp["date_calculated"] = datetime.now().strftime("%Y-%m-%d")

        temp = temp[
            [
                "stock1",
                "stock2",
                "start_date",
                "end_date",
                "date_calculated",
                "max_eigen",
                "trace_statistic",
                "critical_value_max_eigen",
                "critical_value_trace_statistic",
                "cointegrated",
            ]
        ]

        johansen_results = pd.concat([johansen_results, temp])

        johansen_results.drop_duplicates(
            subset=["stock1", "stock2"], keep="last", inplace=True
        )

        for col in johansen_results.columns:
            if "date" in col.lower():
                johansen_results[col] = pd.to_datetime(
                    johansen_results[col], format="mixed"
                )

        johansen_results.to_csv(
            f'{project_path}/Outputs/1_Pairing/Cointegration Test/Johansen/Johansen Test {datetime.now().strftime("%Y-%m-%d")}.csv',
            index=False,
        )

    elif recalculate_johansen == "NEW":

        # List of pairs that have been calculated prior
        calculated_combinations = []
        calculated_combinations = johansen_results[["stock1", "stock2"]].values.tolist()

        new_combinations = new_combinations_list(
            all_combinations, calculated_combinations
        )
        print(f"{len(new_combinations)} new combinations to be calculated")

        # Set Johansen test parameters
        det_order = 0  # No deterministic trend
        k_ar_diff = 1  # Lag order
        significance_level = 0.05

        # Initialize an empty list to store results
        temp = []

        # Perform Johansen cointegration test for all pairs of assets
        for i, j in new_combinations:

            stock1 = i
            stock2 = j
            asset_pair = fin_data3[[stock1, stock2]]
            start_date = min(asset_pair.index[0], asset_pair.index[0])
            end_date = max(asset_pair.index[-1], asset_pair.index[-1])

            # Perform Johansen cointegration test
            result = johansen_cointegration_test(
                asset_pair.values, det_order, k_ar_diff, significance_level
            )

            # Determine if cointegrated
            cointegrated = (
                result["max_eigen"] > result["critical_value_max_eigen"]
            ) and (result["trace_statistic"] > result["critical_value_trace_statistic"])

            # Append results to list
            temp.append(
                {
                    "stock1": stock1,
                    "stock2": stock2,
                    "start_date": start_date,
                    "end_date": end_date,
                    "max_eigen": result["max_eigen"],
                    "trace_statistic": result["trace_statistic"],
                    "critical_value_max_eigen": result["critical_value_max_eigen"],
                    "critical_value_trace_statistic": result[
                        "critical_value_trace_statistic"
                    ],
                    "cointegrated": cointegrated,
                }
            )

        if len(temp) != 0:
            # Create DataFrame from results
            temp = pd.DataFrame(temp)

            temp["date_calculated"] = datetime.now().strftime("%Y-%m-%d")

            temp = temp[
                [
                    "stock1",
                    "stock2",
                    "start_date",
                    "end_date",
                    "date_calculated",
                    "max_eigen",
                    "trace_statistic",
                    "critical_value_max_eigen",
                    "critical_value_trace_statistic",
                    "cointegrated",
                ]
            ]

            johansen_results = pd.concat([johansen_results, temp])

            johansen_results.drop_duplicates(
                subset=["stock1", "stock2"], keep="last", inplace=True
            )

            for col in johansen_results.columns:
                if "date" in col.lower():
                    johansen_results[col] = pd.to_datetime(
                        johansen_results[col], format="mixed"
                    )

            johansen_results.to_csv(
                f'{project_path}/Outputs/1_Pairing/Cointegration Test/Johansen/Johansen Test {datetime.now().strftime("%Y-%m-%d")}.csv',
                index=False,
            )

        else:
            print("\033[1mCointegration results exist for all combinations\033[0m")

    elif recalculate_johansen == "NO":

        print("\033[1mCointegration results loaded\033[0m")

    return johansen_results


############################################################################################################################################################


def pairing(corr_matrix, johansen_results, max_attempts=25):

    paired_df = pd.DataFrame(
        columns=[
            "stock1",
            "stock2",
            "correlation",
            "cointegrated",
        ]
    )

    for stock1 in corr_matrix.columns:
        attempts = 0
        stock2 = None

        while attempts < max_attempts:

            temp_corr_ticker = corr_matrix[stock1].idxmax()

            # Check if this pair is cointegrated in both tests
            if temp_corr_ticker != stock1:
                if (
                    (
                        (
                            johansen_results[johansen_results["cointegrated"] == True][
                                "stock1"
                            ]
                            == stock1
                        )
                        & (
                            johansen_results[johansen_results["cointegrated"] == True][
                                "stock2"
                            ]
                            == temp_corr_ticker
                        )
                    )
                    | (
                        (
                            johansen_results[johansen_results["cointegrated"] == True][
                                "stock1"
                            ]
                            == temp_corr_ticker
                        )
                        & (
                            johansen_results[johansen_results["cointegrated"] == True][
                                "stock2"
                            ]
                            == stock1
                        )
                    )
                ).any():
                    stock2 = temp_corr_ticker
                    break  # Found a pair cointegrated in both tests, exit loop

            # Increment attempts and remove the highest correlation pair from consideration
            attempts += 1
            corr_matrix.at[stock1, temp_corr_ticker] = -1.0  # Mark as visited

        if stock2 is not None:
            correlation_value = corr_matrix.loc[stock1, stock2]
            paired_df = pd.concat(
                [
                    paired_df,
                    pd.DataFrame(
                        {
                            "stock1": [stock1],
                            "stock2": [stock2],
                            "correlation": [correlation_value],
                            "cointegrated": [True],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return paired_df


############################################################################################################################################################


def dev_thresh_z_score(pairs_df, cl_spread, cl_stock, pairs_list, maintain_threshold=0.05):
    """
    Calculate and evaluate z-scores for pairs trading strategies and generate trade signals.

    This function computes z-scores for spreads and individual stocks in pairs trading. It generates trade 
    signals based on specified confidence levels and evaluates the results. A long (or short) position is 
    entered if the spread is above the critical value for the spread (dev_spread_new) and the z-score of the 
    individual stock is below (or above) the negative (or positive) critical value for the stock (dev_stock_new). 
    The position will be maintained until the z-score of the stock moves to below (or above) the critical value 
    minus the `maintain_threshold` (dev_stock_maintain). The function also calculates the adjusted returns for each 
    stock in the pairs based on these trade signals.

    Parameters:
    pairs_df (DataFrame): DataFrame containing pairs data with multi-level columns for spreads and stock prices.
    cl_spread (float): Confidence level for the z-score of the spread. Determines the critical value for spread.
    cl_stock (float): Confidence level for the z-score of individual stocks. Determines the critical value for stocks.
    pairs_list (list): List of tuples, each containing two stock names for the pairs trading strategy.
    maintain_threshold (float): Threshold to adjust the critical value for maintaining positions. Default is 0.05.
    """

    global pairs1_z_score

    # devs_spread = critical value for spread
    # devs_stocks = critical value for stocks

    # Calculate the critical values for spread and stocks based on confidence levels
    devs_spread_new = norm.ppf(cl_spread)
    devs_stock_new = norm.ppf(cl_stock)

    devs_stock_maintain = norm.ppf(cl_stock-maintain_threshold)

    pairs1_z_score = pairs_df

    for stock1, stock2 in pairs_list:
        spread_mean = pairs1_z_score[(f"{stock1}_{stock2}", "spread")].mean()
        spread_std = pairs1_z_score[(f"{stock1}_{stock2}", "spread")].std()
        stock1_mean = pairs1_z_score[(f"{stock1}_{stock2}", f"{stock1}")].mean()
        stock1_std = pairs1_z_score[(f"{stock1}_{stock2}", f"{stock1}")].std()
        stock2_mean = pairs1_z_score[(f"{stock1}_{stock2}", f"{stock2}")].mean()
        stock2_std = pairs1_z_score[(f"{stock1}_{stock2}", f"{stock2}")].std()

        # Calculate Z-scores for spread and individual stocks
        pairs1_z_score[(f"{stock1}_{stock2}", "z_score_spread")] = (
            pairs1_z_score[(f"{stock1}_{stock2}", "spread")] - spread_mean
        ) / spread_std
        pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock1}")] = (
            pairs1_z_score[(f"{stock1}_{stock2}", f"{stock1}")] - stock1_mean
        ) / stock1_std
        pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock2}")] = (
            pairs1_z_score[(f"{stock1}_{stock2}", f"{stock2}")] - stock2_mean
        ) / stock2_std

        # Define conditions for generating trade signals

# New Trade Signals

        conditions1_new = [
            (pairs1_z_score[(f"{stock1}_{stock2}", "z_score_spread")] > devs_spread_new) 
            & (pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock1}")] < -devs_stock_new),
            
            (pairs1_z_score[(f"{stock1}_{stock2}", "z_score_spread")] > devs_spread_new) 
            & (pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock1}")] > devs_stock_new),
        ]

        conditions2_new = [
            (pairs1_z_score[(f"{stock1}_{stock2}", "z_score_spread")] > devs_spread_new) 
            & ((pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock2}")]) < -devs_stock_new),
            
            (pairs1_z_score[(f"{stock1}_{stock2}", "z_score_spread")] > devs_spread_new) 
            & (pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock2}")] > devs_stock_new),
        ]

        choices = ["Long", "Short"]

        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")] = np.select(
            conditions1_new, choices, default=""
        )
        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")] = np.select(
            conditions2_new, choices, default=""
        )


# Hold the position until the Z-score of the stock is above/below the CI - 0.05
        
        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}_maintain")] = np.where(
            (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")].shift(1) != '') 
                & ((pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock1}")] < -devs_stock_maintain) | (pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock1}")] > devs_stock_maintain)),
            pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")].shift(1),
            '')
            
        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}_maintain")] = np.where(
            (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")].shift(1) != '') 
                & ((pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock2}")] < -devs_stock_maintain) | (pairs1_z_score[(f"{stock1}_{stock2}", f"z_score_{stock2}")] > devs_stock_maintain)),
            pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")].shift(1),
            '')   
        
        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")] = pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}_maintain")].replace([np.nan, np.inf, -np.inf], '')
        pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")] = pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}_maintain")].replace([np.nan, np.inf, -np.inf], '')
        
        pairs1_z_score = pairs1_z_score.drop(columns=[(f"{stock1}_{stock2}", f"position_{stock1}_maintain"), (f"{stock1}_{stock2}", f"position_{stock2}_maintain")])

        # Ensure trade signals are opposite and in pairs
        mask1 = np.where(
            pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")]
            == pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")],
            True,
            False,
        )
  
        mask2 = np.where(
            (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")] != "")
            & (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")] == "")
            | (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")] == "")
            & (pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")] != ""),
            True,
            False,
        )

        mark1_2 = mask1 | mask2

        pairs1_z_score.loc[
            mark1_2,
            [
                (f"{stock1}_{stock2}", f"position_{stock1}"),
                (f"{stock1}_{stock2}", f"position_{stock2}"),
            ],
        ] = ""
        
        # Daily Returns
        ## Currency 
        position_map = {"Long": 1, "Short": -1, "Blank": 0}
        pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock1}_usd")] = pairs1_z_score[
            (f"{stock1}_{stock2}", f"{stock1}")].diff().shift(-1) * pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock1}")].map(position_map)
        
        pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock2}_usd")] = pairs1_z_score[
            (f"{stock1}_{stock2}", f"{stock2}")].diff().shift(-1) * pairs1_z_score[(f"{stock1}_{stock2}", f"position_{stock2}")].map(position_map)
        
        pairs1_z_score[(f"{stock1}_{stock2}", "tot_return_usd")] = (
            pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock1}_usd")]
            + pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock2}_usd")])
        
        ## Percent
        pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock1}_pct")] = pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock1}_usd")] / pairs1_z_score[(f"{stock1}_{stock2}", f"{stock1}")]
        pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock2}_pct")] = pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock2}_usd")] / pairs1_z_score[(f"{stock1}_{stock2}", f"{stock2}")]
       
        pairs1_z_score[(f"{stock1}_{stock2}", "tot_return_pct")] = (
            pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock1}_pct")]
            + pairs1_z_score[(f"{stock1}_{stock2}", f"return_{stock2}_pct")])

    
    pairs1_z_score.sort_index(level=0, axis=1, inplace=True)
    print(
        f"Total Absolute Return (Z-Score Spread: {cl_spread}, Stock: {cl_stock},  Maintain: {maintain_threshold}): {pairs1_z_score.loc[:, pairs1_z_score.columns.get_level_values(1) == 'tot_return_usd'].sum().sum()}"
    )
    
    return pairs1_z_score

############################################################################################################################################################

# Sample implementation of the calculate_average_days_live function
def calculate_average_days_live(stock_df):

    """
    Calculate the average number of days a trade is live for a given stock.

    Parameters:
    stock_df (pd.DataFrame): DataFrame containing the 'Date' and 'Position' columns for a specific stock.
    
    Returns:
    float: The average number of days a trade is live. Returns 0 if there are no live trades.
    """
    
    live_trade_periods = []
    start_date = None
    
    for index, row in stock_df.iterrows():
        position = row['Position']
        date = row['Date']
        
        if position and start_date is None:
            start_date = date
        elif not position and start_date is not None:
            end_date = date
            live_trade_periods.append((end_date - start_date).days)
            start_date = None
        elif position and start_date is not None and position != stock_df.loc[index - 1, 'Position']:
            end_date = date
            live_trade_periods.append((end_date - start_date).days)
            start_date = date

    # If a trade is live at the end of the period, close it at the last date
    if start_date is not None:
        end_date = stock_df['Date'].iloc[-1]
        live_trade_periods.append((end_date - start_date).days + 1)
    
    average_days_live = sum(live_trade_periods) / len(live_trade_periods) if live_trade_periods else 0
    return average_days_live

############################################################################################################################################################

def curve_metrics(pairs_list, df):
    """
    Calculate additional trading metrics for a list of stock pairs from the provided DataFrame.

    Parameters:
    pairs_list (list of tuples): List of tuples where each tuple contains two stock names.
    df (pd.DataFrame): DataFrame containing the trading data. It must include columns for each stock's position and total return.

    Returns:
    pd.DataFrame: DataFrame containing calculated metrics for each stock pair, including:
        - Total Return ($): The sum of all trade returns.
        - CAGR (%): Compound Annual Growth Rate, representing the annualized return over the period.
          Calculated as: \(((\text{Ending Value} / \text{Beginning Value})^{(1 / \text{Number of Years})}) - 1\) * 100.
        - Annual Volatility (%): The standard deviation of the returns, annualized.
          Calculated as: \(\text{Standard Deviation of Daily Returns} \times \sqrt{252}\) * 100.
        - Trades per Year: The average number of trades executed per year.
          Calculated as: \(\text{Total Number of Trades} / \text{Number of Years}\).
        - Max Daily Drawdown ($): The maximum drop from a peak to a trough in daily returns.
        - Max Drawdown Duration (days): The maximum duration (in days) of the drawdown period.
    """
    df = df.reset_index()
    results = pd.DataFrame()

    for stock1, stock2 in pairs_list:
        # Replace None with '' in the required columns

        # Create a temp DataFrame for analysis
        temp = df[[f"{stock1}_{stock2}"]]
        temp = temp.droplevel(level=0, axis=1).reset_index()
        
        # Create a temp DataFrame for analysis
        temp = df[[f"{stock1}_{stock2}"]]
        temp = temp.droplevel(level=0, axis=1)
        
        # Add a 'Return' column
        temp['Return'] = temp['tot_return_usd']
        temp['Date'] = df['Date']
        
        # Total Return
        total_return = temp['Return'].sum()
        
        # CAGR (Compound Annual Growth Rate)
        start_value = temp['Return'].iloc[0] if len(temp) > 0 else 1
        end_value = temp['Return'].iloc[-1] if len(temp) > 0 else 1
        num_years = (temp['Date'].max() - temp['Date'].min()).days / 365.25
        cagr = ((end_value / start_value) ** (1 / num_years) - 1) * 100 if num_years > 0 else 0
        
        # Annual Volatility
        daily_returns = temp['Return']
        daily_volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Trades per Year
        num_trades = len(temp)
        num_years = (temp['Date'].max() - temp['Date'].min()).days / 365.25
        trades_per_year = num_trades / num_years if num_years > 0 else 0

        # Store results
        temp_results = pd.DataFrame(
            {
                f"{stock1}_{stock2}": [
                    total_return,
                    cagr,
                    daily_volatility,
                    trades_per_year,
                ]
            },
            index=[
                "Total Return ($)",
                "CAGR (%)",
                "Annual Volatility (%)",
                "Trades per Year",
            ],
        )

        results = pd.concat([temp_results, results], axis=1)

    summary_pairs = results.T
    summary_pairs.sort_values(by="Total Return ($)", ascending=False, inplace=True)
    
    return summary_pairs

############################################################################################################################################################

def trade_metrics(pairs_list, df):
    
    """
    Calculate various trading metrics for a list of stock pairs from the provided DataFrame.

    Parameters:
    pairs_list (list of tuples): List of tuples where each tuple contains two stock names.
    df (pd.DataFrame): DataFrame containing the trading data. It must include columns for each stock's position and total return.

    Returns:
    pd.DataFrame: DataFrame containing calculated metrics for each stock pair, including:
        - Trade Winning Percentage (%): The percentage of trades with a positive return. 
          Calculated as: (Number of Winning Trades / Total Number of Trades) * 100.
        - Best Trade ($): The highest return achieved in a single trade.
        - Worst Trade ($): The lowest return achieved in a single trade.
        - Average Win ($): The average return for trades that are profitable.
          Calculated as: Sum of Positive Trade Returns / Number of Winning Trades.
        - Average Loss ($): The average return for trades that result in a loss.
          Calculated as: Sum of Negative Trade Returns / Number of Losing Trades.
        - Average Trade Return ($): The average return across all trades.
          Calculated as: Sum of All Trade Returns / Total Number of Trades.
        - Total Return ($): The sum of returns for all trades.
        - Number of Trades: The total count of trades executed, including both profitable and losing trades.
        - Average Days in Trade: The average number of days that a trade remains open before it is closed.
    """
    
    results = pd.DataFrame()

    for stock1, stock2 in pairs_list:
        temp = df[[f"{stock1}_{stock2}"]]
        temp = temp.droplevel(level=0, axis=1).reset_index()

        try:
            # Calculate Trade Winning %
            trade_winning_percentage = len(temp[temp["tot_return_usd"] > 0]) / len(
                temp[temp[f"position_{stock1}"] != ""]
            )
        except:
            trade_winning_percentage = 0

        # Calculate Best Trade
        best_trade = temp["tot_return_usd"].max()

        # Calculate Worst Trade
        worst_trade = temp["tot_return_usd"].min()

        # Calculate Average Win
        average_win = temp[temp["tot_return_usd"] > 0]["tot_return_usd"].mean()

        # Calculate Average Loss
        average_loss = temp[temp["tot_return_usd"] < 0]["tot_return_usd"].mean()

        # Calculate Average Trade Return
        average_trade_return = temp["tot_return_usd"].mean()

        # Total Trade Return
        total_trade_return = temp["tot_return_usd"].sum()

        # Calculate number of 'paired' positions
        no_of_trades = len(temp[temp[f"position_{stock1}"] != ""])
        
        # Calculate Average Days Live
        stock_df = temp[['Date', f"position_{stock1}"]].copy()
        stock_df.columns = ['Date', 'Position']
        average_days_live = calculate_average_days_live(stock_df)

        temp_results = pd.DataFrame(
            {
                f"{stock1}_{stock2}": [
                    trade_winning_percentage,
                    best_trade,
                    worst_trade,
                    average_win,
                    average_loss,
                    average_trade_return,
                    total_trade_return,
                    no_of_trades,
                    average_days_live
                ]
            },
            index=[
                "Trade Winning Percentage (%)",
                "Best Trade ($)",
                "Worst Trade ($)",
                "Average Win ($)",
                "Average Loss ($)",
                "Average Trade Return ($)",
                "Total Return ($)",
                "Number of Trades",
                'Avg. Days in Trade'
            ],
        )

        results = pd.concat([temp_results, results], axis=1)

    summary_pairs = results.T
    summary_pairs.sort_values(by="Total Return ($)", inplace=True)
        
    return summary_pairs

############################################################################################################################################################

def time_metrics(pairs_list, df):
    """
    Calculate various monthly trading metrics for a list of stock pairs from the provided DataFrame.

    Parameters:
    pairs_list (list of tuples): List of tuples where each tuple contains two stock names.
    df (pd.DataFrame): DataFrame containing the trading data. It must include columns for each stock's position and total return.

    Returns:
    pd.DataFrame: DataFrame containing calculated monthly metrics for each stock pair, including:
        - Winning Months (%)
        - Average Return for Winning Month ($)
        - Average Return for Losing Month ($)
        - Best Month (% Return)
        - Worst Month (% Return)
    """
    df = df.reset_index()
    results = pd.DataFrame()

    for stock1, stock2 in pairs_list:
        # Replace None with '' in the required columns

        
        # Create a temp DataFrame for analysis
        temp = df[[f"{stock1}_{stock2}"]]
        temp = temp.droplevel(level=0, axis=1).reset_index()
        
        # Add month and year columns
        temp['Month'] = df['Date'].dt.to_period('M')
        temp['Year'] = df['Date'].dt.to_period('Y')
        temp['Return'] = temp['tot_return_usd']

        # Aggregate returns by month
        monthly_returns = temp.groupby('Month')['Return'].sum()

        # Aggregate returns by year
        yearly_returns = temp.groupby('Year')['Return'].sum()

        # Calculate monthly metrics
        num_months = len(monthly_returns)
        num_winning_months = len(monthly_returns[monthly_returns > 0])
        num_losing_months = len(monthly_returns[monthly_returns < 0])
        
        winning_months_percentage = (num_winning_months / num_months) * 100 if num_months > 0 else 0
        
        avg_return_winning_month = monthly_returns[monthly_returns > 0].mean() if num_winning_months > 0 else 0
        avg_return_losing_month = monthly_returns[monthly_returns < 0].mean() if num_losing_months > 0 else 0
        
        best_month_return = monthly_returns.max()
        worst_month_return = monthly_returns.min()

        # Calculate yearly metrics
        num_years = len(yearly_returns)
        num_winning_years = len(yearly_returns[yearly_returns > 0])
        num_losing_years = len(yearly_returns[yearly_returns < 0])
        
        winning_years_percentage = (num_winning_years / num_years) * 100 if num_years > 0 else 0
        
        avg_return_winning_year = yearly_returns[yearly_returns > 0].mean() if num_winning_years > 0 else 0
        avg_return_losing_year = yearly_returns[yearly_returns < 0].mean() if num_losing_years > 0 else 0
        
        best_year_return = yearly_returns.max()
        worst_year_return = yearly_returns.min()

        # Store results
        temp_results = pd.DataFrame(
            {
                f"{stock1}_{stock2}": [
                    winning_years_percentage,
                    avg_return_winning_year,
                    avg_return_losing_year,
                    best_year_return,
                    worst_year_return,
                    winning_months_percentage,
                    avg_return_winning_month,
                    avg_return_losing_month,
                    best_month_return,
                    worst_month_return
                ]
            },
            index=[
                "Winning Years (%)",
                "Average Return for Winning Year ($)",
                "Average Return for Losing Year ($)",
                "Best Year (% Return)",
                "Worst Year (% Return)",
                "Winning Months (%)",
                "Average Return for Winning Month ($)",
                "Average Return for Losing Month ($)",
                "Best Month (% Return)",
                "Worst Month (% Return)"
            ],
        )

        results = pd.concat([temp_results, results], axis=1)

    summary_pairs = results.T
    summary_pairs.sort_values(by="Best Year (% Return)", ascending=False, inplace=True)
    
    return summary_pairs