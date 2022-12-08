import os

import numpy as np
import pandas as pd
import datetime
import logging
import itertools
import datetime

from finrl import config
from finrl import config_tickers
from finrl.data.preprocessor.preprocessors import FeatureEngineer
from finrl.data.preprocessor.yahoodownloader import YahooDownloader

logger = logging.getLogger(__name__)


class DatasetFactory:

    def __init__(self, ticker: list, ticker_list_name: str):

        self.ticker = ticker
        self.ticker_list_name = ticker_list_name
        self.start_date = config.TRAIN_START_DATE
        self.end_date = config.TRADE_END_DATE
        self.interval = config.DATA_INTERVAL

        # Directory to save the raw ticker data to
        self.raw_dir = config.RAW_DATA_SAVE_DIR
        # Directory to save the preprocessed dataset
        self.name = '{}_{}_{}'.format(self.ticker_list_name, self.start_date, self.end_date)
        self.prepro_filename = os.path.join(config.PREPRO_DATA_SAVE_DIR, '{}'.format(self.name))

    def load_and_update_raw_data(self, raw_data_file):

        df = pd.read_pickle(raw_data_file)
        # Check if Update is neccessary
        last_date = df.iloc[-1]['date']
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
        final_target_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        diff = (final_target_date - last_date).days

        if diff > 4:
            logger.info("Doing data update (LastDate: '{}', Target date: '{}')".format(last_date, final_target_date))
            update_df = YahooDownloader(
                start_date=last_date.strftime("%Y-%m-%d"),
                end_date=final_target_date.strftime("%Y-%m-%d"),
                ticker_list=self.ticker,
            ).fetch_data()
            df = pd.concat(df, update_df)
            logger.info("Last date of updated df: '{}'. Saving as '{}'".format(df.iloc[-1]['date'], self.raw_filename))
            df.to_pickle(self.raw_filename)
        else:
            logger.info("Raw data is recent enough for not doing update".format(raw_data_file))

        return df

    def load_preprocessed_dataset(self):
        return pd.read_pickle(self.prepro_filename)


    def _check_raw_data(self, t):
        # Try to find existing ticker raw data
        data = [os.path.join(config.RAW_DATA_SAVE_DIR, d) for d in os.listdir(config.RAW_DATA_SAVE_DIR) if
                d.startswith(t + '_')]
        if len(data) > 1:
            logger.error("Found multiple raw files for ticker '{}'. Investigate this".format(t))
            exit()
        elif len(data) == 0:
            logger.info("Found no existing raw data for ticker '{}'".format(t))
            return None
        else:
            logger.info("Loading raw_data '{}' for updating".format(data[0]))
            return data[0]

    def break_up_date_range(self, intv_size=50):
        start = datetime.datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        final_end = end
        n_intervals = int(np.ceil((end - start).days/intv_size))
        all_intervals = []
        for i in range(n_intervals):
            if i > 0:
                start += datetime.timedelta(days=intv_size + 1)
            end = start + datetime.timedelta(days=intv_size)
            if end > final_end:
                all_intervals.append((start.strftime("%Y-%m-%d"), final_end.strftime("%Y-%m-%d")))
                return all_intervals

            all_intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))


    def download_new_ticker_data(self, t):
        logger.info("Downlaoding new Dataset for ticker '{}'".format(self.ticker_list_name))

        if self.interval in ["1m","2m","5m","15m","30m","60m","90m","1h"]:

            dr = self.break_up_date_range()
            full_df = pd.DataFrame()
            for start_end in dr:

                df = YahooDownloader(
                    start_date=start_end[0],
                    end_date=start_end[1],
                    ticker_list=self.ticker,
                    interval=self.interval
                ).fetch_data()
                tmp_out = os.path.join(self.raw_dir, "tmp", '{}_{}_{}'.format(t, start_end[0], start_end[1]))
                logger.info("Saving new tmp data as '{}'".format(self.raw_dir))
                df.to_pickle(tmp_out)
                full_df = pd.concat([full_df, df])

            out_name = os.path.join(self.raw_dir, '{}_{}_{}'.format(t, self.start_date, self.end_date))
            logger.info("Saving new {} data as '{}'".format(self.interval, out_name))
            df.to_pickle(out_name)

            exit()

        df = YahooDownloader(
            start_date=self.start_date,
            end_date=self.end_date,
            ticker_list=self.ticker,
            interval=self.interval
        ).fetch_data()
        logger.info("Saving new datset as '{}'".format(self.raw_filename))
        df.to_pickle(self.raw_filename)

        return df

    def create_dataset(self):
        # Check if a preprocessed_dataframe_exists
        if os.path.isfile(self.prepro_filename):
            logger.info("Loading preprocessed dataset '{}'".format(self.prepro_filename))
            df = self.load_preprocessed_dataset()
        else:
            logger.info("No preprocessed dataset found. Assembling from Raw data")

            for ticker in self.ticker:

                # Check if raw data for ticker typpe eixsts and extent if so
                raw_data = self._check_raw_data(ticker)
                if raw_data is not None:
                    df = self.load_and_update_raw_data(raw_data)
                else:
                    df = self.download_new_ticker_data(ticker)

            logger.info("Preprocessing raw data")
            df = self.preprocess_data(df)
            logger.info("Saving preprocessed data as '{}'".format(self.prepro_filename))
            df.to_pickle(self.prepro_filename)

        logger.info("Using Ticker: {}".format(self.ticker))
        logger.info("Full dataset shape: {}".format(df.shape))
        df.sort_values(["date", "tic"], ignore_index=True).head()

        return df


    def preprocess_data(self, df):
        # """
        # # Part 4: Preprocess Data
        # Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
        # * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
        # * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
        # """

        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.INDICATORS,
            use_vix=True,
            use_turbulence=True,
            user_defined_feature=False,
        )

        processed = fe.preprocess_data(df)

        list_ticker = processed["tic"].unique().tolist()
        list_date = list(pd.date_range(processed["date"].min(), processed["date"].max()).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(processed, on=["date", "tic"],
                                                                                  how="left")
        processed_full = processed_full[processed_full["date"].isin(processed["date"])]
        processed_full = processed_full.sort_values(["date", "tic"])

        processed_full = processed_full.fillna(0)
        processed_full.sort_values(["date", "tic"], ignore_index=True).head(10)

        return processed_full

    def create_baseline(self):

        # Check if raw data for ticker typpe eixsts and extent if so
        raw_data = self._check_raw_data()
        if raw_data is not None:
            df = self.load_and_update_raw_data(raw_data)
        else:
            df = self.download_new_dataset()

        return df


class Dataset:

    def __init__(self, name, start, end):

        self.name = name


