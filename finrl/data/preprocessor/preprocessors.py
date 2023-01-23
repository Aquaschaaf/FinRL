from __future__ import annotations

import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from finrl import config
from finrl.data.preprocessor.yahoodownloader import YahooDownloader

import logging
logger = logging.getLogger(__name__)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    # _data = pd.read_csv(f"{config.DATASET_DIR}/{file_name}")
    _data = pd.read_csv(file_name)
    return _data


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def convert_to_datetime(time):
    time_fmt = "%Y-%m-%dT%H:%M:%S"
    if isinstance(time, str):
        return datetime.datetime.strptime(time, time_fmt)


class FeatureEngineer:
    """Provides methods for preprocessing the stock price data

    Attributes
    ----------
        use_technical_indicator : boolean
            we technical indicator or not
        tech_indicator_list : list
            a list of technical indicator names (modified from neofinrl_config.py)
        use_turbulence : boolean
            use turbulence index or not
        user_defined_feature:boolean
            use user defined features or not

    Methods
    -------
    preprocess_data()
        main method to do the feature engineering

    """

    def __init__(
        self,
        use_technical_indicator=True,
        tech_indicator_list=config.INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
        clean_strategy="remove_cols"
    ):
        self.use_technical_indicator = use_technical_indicator
        self.tech_indicator_list = tech_indicator_list
        self.use_vix = use_vix
        self.use_turbulence = use_turbulence
        self.user_defined_feature = user_defined_feature
        self.clean_strategy = clean_strategy

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # clean data
        df = self.clean_data(df, self.clean_strategy)

        # add technical indicators using stockstats
        if self.use_technical_indicator:
            df = self.add_technical_indicator(df)
            print("Successfully added technical indicators")

        # add vix for multiple stock - Deprectaed - moved to dataset tickers
        if self.use_vix:
            df = self.add_vix(df)
            print("Successfully added vix")

        # add turbulence index for multiple stock
        if self.use_turbulence:
            df = self.add_turbulence(df)
            print("Successfully added turbulence index")

        # add user defined feature
        if self.user_defined_feature:
            df = self.add_user_defined_feature(df)
            print("Successfully added user defined features")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="ffill").fillna(method="bfill")
        return df

    def clean_data(self, data, strategy="remove_cols"):
        """
        clean the raw data
        deal with missing values
        reasons: stocks could be delisted, not incorporated at the time step
        :param data: (df) pandas dataframe
        :param strategy: (string) the strategy to apply
        :return: (df) pandas dataframe
        """
        logger.info("Cleaning data with startegy {}".format(strategy))
        df = data.copy()
        df = df.sort_values(["datetime", "tic"], ignore_index=True)
        # df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]

        if strategy == "remove_cols":
            merged_closes = df.pivot_table(index="datetime", columns="tic", values="close")
            # merged_closes = df.pivot_table(index="date", columns="tic", values="close")
            nans = merged_closes.isna().any(axis=1)
            merged_closes = merged_closes.dropna(axis=1)
            tics = merged_closes.columns
            df = df[df.tic.isin(tics)]
        elif strategy == "remove_rows":
            logger.warning("DATA Might contain non-sequential intervals due to data cleaning startegy!")
            df = df.dropna(axis = 0, how = 'all')

        # df = data.copy()
        # list_ticker = df["tic"].unique().tolist()
        # only apply to daily level data, need to fix for minute level
        # list_date = list(pd.date_range(df['date'].min(),df['date'].max()).astype(str))
        # combination = list(itertools.product(list_date,list_ticker))

        # df_full = pd.DataFrame(combination,columns=["date","tic"]).merge(df,on=["date","tic"],how="left")
        # df_full = df_full[df_full['date'].isin(df['date'])]
        # df_full = df_full.sort_values(['date','tic'])
        # df_full = df_full.fillna(0)
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "datetime"])
        # df = df.sort_values(by=["tic", "date"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        logger.info("Adding Indicators to DataFrame.")
        for indicator in tqdm(self.tech_indicator_list):
            indicator_df = pd.DataFrame()
            logging.info("Addinng {}".format(indicator))

            for i in range(len(unique_ticker)):
                if unique_ticker[i] == "^VIX":
                    logger.info("Skipping VIX for technical indicator extraction")
                    continue

                if indicator == "normalized_close":
                    min = df[df.tic == unique_ticker[i]].close.expanding(min_periods=10).min()
                    max = df[df.tic == unique_ticker[i]].close.expanding(min_periods=10).max()

                    temp_indicator = ((df[df.tic == unique_ticker[i]].close - min) / (max - min)).to_frame()
                    temp_indicator.columns = ['{}'.format(indicator)]
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["datetime"] = df[df.tic == unique_ticker[i]]["datetime"].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator])

                else:
                    # Calculate indicator based on StockDataframe
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["datetime"] = df[df.tic == unique_ticker[i]]["datetime"].to_list()
                    # temp_indicator["date"] = df[df.tic == unique_ticker[i]]["datetime"].to_list()
                    indicator_df = pd.concat([indicator_df, temp_indicator])
                    # indicator_df = indicator_df.append(temp_indicator, ignore_index=True)

            try:
                df = df.merge(indicator_df[["tic", "datetime", indicator]], on=["tic", "datetime"], how="left")
                # df = df.merge(indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left")
            except:
                logger.error("Failed to merge data for indicator: {}".format(indicator))
                exit()
        df = df.sort_values(by=["datetime", "tic"])
        # df = df.sort_values(by=["date", "tic"])
        return df
        # df = data.set_index(['date','tic']).sort_index()
        # df = df.join(df.groupby(level=0, group_keys=False).apply(lambda x, y: Sdf.retype(x)[y], y=self.tech_indicator_list))
        # return df.reset_index()

    def add_user_defined_feature(self, data):
        """
         add user defined features
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df["daily_return"] = df.close.pct_change(1)
        # df['return_lag_1']=df.close.pct_change(2)
        # df['return_lag_2']=df.close.pct_change(3)
        # df['return_lag_3']=df.close.pct_change(4)
        # df['return_lag_4']=df.close.pct_change(5)
        return df

    def add_vix(self, data):
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        # df_vix = YahooDownloader(
        #     start_date=df.date.min(), end_date=df.date.max(), ticker_list=["^VIX"]
        # ).fetch_data()

        # Extract Vix rows from dataframe
        vix_msk = df["tic"] == "^VIX"
        df_vix = df[vix_msk]
        df = df[~vix_msk]

        vix = df_vix[["datetime", "close"]]
        # vix = df_vix[["date", "close"]]
        vix.columns = ["datetime", "vix"]
        # vix.columns = ["date", "vix"]

        df = df.merge(vix, on="datetime")
        # df = df.merge(vix, on="date")
        df = df.sort_values(["datetime", "tic"]).reset_index(drop=True)
        # df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="datetime")
        # df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["datetime", "tic"]).reset_index(drop=True)
        # df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="datetime", columns="tic", values="close")
        # df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 252
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 252])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
        try:
            turbulence_index = pd.DataFrame({"datetime": df_price_pivot.index, "turbulence": turbulence_index})
            # turbulence_index = pd.DataFrame({"date": df_price_pivot.index, "turbulence": turbulence_index})
        except ValueError:
            raise Exception("Turbulence information could not be added.")
        return turbulence_index



