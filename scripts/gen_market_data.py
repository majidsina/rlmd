"""
title:                          gen_market_data.py
usage:                          python gen_market_data.py
python version:                 3.10
pandas-datareader version:      0.10

code style:                     black==22.3
import style:                   isort==5.10

copyright (C):                  https://www.gnu.org/licenses/agpl-3.0.en.html
author:                         J. S. Grewal (2022)
email:                          <rg (_] public [at} proton {dot) me>
linkedin:                       https://www.linkedin.com/in/rajagrewal
website:                        https://www.github.com/rajabinks

Description:
    Collects market data from online sources and creates NumPy arrays ready for
    agent learning. All data is remotely (and freely) obtained using pandas-datareader
    following https://pydata.github.io/pandas-datareader/remote_data.html.

    Historical data for major indices, commodities, and currencies is obtained from
    Stooq at https://stooq.com/. Note not every symbol can be utilised, all must be
    individually checked to determine feasibility.

    Occasionally will receive "SymbolWarning: Failed to read symbol" from Stooq API,
    running the script again usually fixes this but might not probably.

Instructions:
    1. Select appropriate start and end date for data for all assets with daily data
       sampling frequency.
    2. Enter into the dictionary the obtained Stooq symbols for desired assets and
       place them in lisst following the naming scheme.
    3. Running file will scrape data and place it in a directory containing pickle
       and csv files, along with a cleaned NumPy array.

Stooq - Symbols and Data Availability:
    ^SPX: S&P 500                       https://stooq.com/q/d/?s=^spx
    ^DJI: Dow Jones Industrial 30       https://stooq.com/q/d/?s=^dji
    ^NDX: Nasdaq 100                    https://stooq.com/q/d/?s=^ndx

    GC.F: Gold - COMEX                  https://stooq.com/q/d/?s=gc.f
    SI.F: Silver - COMEX                https://stooq.com/q/d/?s=si.f
    HG.F: High Grade Copper - COMEX     https://stooq.com/q/d/?s=hg.f
    PL.F: Platinum - NYMEX              https://stooq.com/q/d/?s=pf.f
    PA.F: Palladium - NYMEX             https://stooq.com/q/d/?s=pa.f

    CL.F: Crude Oil WTI - NYMEX         https://stooq.com/q/d/?s=cl.f
    RB.F: Gasoline RBOB - NYMEX         https://stooq.com/q/d/?s=rb.f

    LS.F: Lumber - CME                  https://stooq.com/q/d/?s=ls.f
    LE.F: Live Cattle - CME             https://stooq.com/q/d/?s=le.f
    KC.F: Coffee - ICE                  https://stooq.com/q/d/?s=kc.f
    OJ.F: Orange Juice - ICE            https://stooq.com/q/d/?s=oj.f

    ^ = index value
    .C = cash
    .F = front month futures
"""

import sys

sys.path.append("./")

import os

import numpy as np
import numpy.typing as npt
import pandas as pd
import pandas_datareader.data as pdr

NDArrayFloat = npt.NDArray[np.float_]

from tests.test_input_extra import market_data_tests

"""
    ********************************************************************************
    AS OF THE TIMING OF THIS MOST RECENT COMMIT, STOOQ NO LONGER ALLOWS THE
    DOWNLOADING OF COMMODITIES DATA AND SO THEY WILL RETURN AN ERROR.
    https://github.com/pydata/pandas-datareader/issues/925
    https://www.github.com/rajabinks/stooq-commodities
    ********************************************************************************
"""

# common starting/endiing dates for daily data collection for all assets
start: str = "1985-10-01"
end: str = "2022-02-10"

# fmt: off

stooq: dict = {
    # pairs for data saving and assets to be included
    # market_id: [market_name, included assets (List[str])]

    "mkt0": ["snp", ["^SPX"]],

    "mkt1": ["usei", ["^SPX", "^DJI", "^NDX"]],

    "mkt2": ["dji", ["^SPX", "^DJI", "^NDX",
                     "AAPL.US", "AMGN.US", "AXP.US", "BA.US", "CAT.US", "CVX.US",
                     "DIS.US", "HD.US", "IBM.US", "INTC.US", "JNJ.US", "JPM.US",
                     "KO.US", "MCD.US", "MMM.US", "MRK.US", "MSFT.US", "NKE.US",
                     "PFE.US", "PG.US", "VZ.US", "WBA.US", "WMT.US",
                     "CSCO.US", "UNH.US",                              # starts 1990
                    #   "CRM.US", "DOW.US", "GS.US", "TRV.US", "V.US"    # very little data
                    ]],

    # unable to update commodities data from Stooq

    # "mkt3": ["minor", ["^SPX", "^DJI", "^NDX",
    #                    "GC.F", "SI.F",
    #                    "CL.F"
    #                    ]],

    # "mkt4": ["medium", ["^SPX", "^DJI", "^NDX",
    #                     "GC.F", "SI.F", "HG.F", "PL.F",
    #                     "CL.F",
    #                     "LS.F"
    #                     ]],

    # "mkt5": ["major", ["^SPX", "^DJI", "^NDX",
    #                    "GC.F", "SI.F", "HG.F", "PL.F", "PA.F",
    #                    "CL.F", "RB.F",
    #                    "LS.F", "LE.F", "KC.F", "OJ.F"
    #                    ]],

    # "mkt6": ["full", ["^SPX", "^DJI", "^NDX",
    #                   "GC.F", "SI.F", "HG.F", "PL.F", "PA.F",
    #                   "CL.F", "RB.F",
    #                   "LS.F", "LE.F", "KC.F", "OJ.F",
    #                   "AAPL.US", "AXP.US", "BA.US", "CAT.US", "CVX.US",
    #                   "DIS.US", "HD.US", "IBM.US", "INTC.US", "JNJ.US", "JPM.US",
    #                   "KO.US", "MCD.US", "MMM.US", "MRK.US", "MSFT.US", "NKE.US",
    #                   "PFE.US", "PG.US", "RTX.US", "VZ.US", "WBA.US", "WMT.US", "XOM.US"
    #                   "CSCO.US", "UNH.US",                   # starts 1990
    #                #    "DOW.US", "GS.US", "TRV.US", "V.US"    # very little data
    #                 ]],
    }

# fmt: on


def dataframe_to_array(
    market_data: pd.DataFrame, price_type: str, volume: bool
) -> NDArrayFloat:
    """
    Converts pandas dataframe to cleaned numpy array by extracting relevant prices.

    Parameters:
        markert_data: raw dataframe generated by pandas_datareader from remote source
        price_type: "Open", "High", "Low", or "Close" prices for the time step
        volume: whether to include volume

    Returns:
        prices: cleaned array of asset prices of a given type
    """
    market = market_data[str(price_type).capitalize()]

    # remove all rows with missing values
    market = market.dropna()

    # format time ordering if needed (earliest data point is at index 0)
    if market.index[0] > market.index[-1]:
        market = market[::-1]

    n_assets, n_days = market.columns.shape[0], market.index.shape[0]

    prices = np.empty((n_days, n_assets), dtype=np.float64)

    a = 0
    for asset in market.columns:
        prices[:, a] = market[str(asset)]
        a += 1

    # placeholder for building volume functionality
    if volume == True:
        market_and_volumes = market_data[[str(price_type).capitalize(), "Volume"]]
        prices_and_volumes = np.empty((n_days, n_assets * 2), dtype=np.float64)

    return prices


if __name__ == "__main__":

    # directory for saving market prices dataframes, csvs, and arrays
    path = "./tools/market_data/"
    # market price type (Open, High, Low, or Close)
    price_type = "Close"

    # CONDUCT TESTS
    market_data_tests(start, end, stooq, path, price_type)

    if not os.path.exists(path):
        os.makedirs(path)

    for x in stooq:
        name = "stooq_" + str(stooq[str(x)][0])
        assets = stooq[str(x)][1]

        scraped_data = pdr.get_data_stooq(assets, start, end)
        scraped_data.to_pickle(path + name + ".pkl")

        market = pd.read_pickle(path + name + ".pkl")

        market.to_csv(path + name + ".csv")

        prices = dataframe_to_array(market, price_type, False)
        np.save(path + name + ".npy", prices)

        print(
            "{}: n_assets = {}, days = {}".format(
                name, prices.shape[1], prices.shape[0]
            )
        )
