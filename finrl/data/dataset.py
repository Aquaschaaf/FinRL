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

        if True:
            self.ticker += ["^VIX"]

        # Directory to save the raw ticker data to
        self.raw_dir = config.RAW_DATA_SAVE_DIR
        # Directory to save the preprocessed dataset
        self.name = '{}_{}_{}'.format(self.ticker_list_name, self.start_date, self.end_date)
        self.prepro_filename = os.path.join(config.PREPRO_DATA_SAVE_DIR, '{}'.format(self.name))

    def load_and_update_raw_data(self, raw_data_file, ticker):

        df = pd.read_pickle(raw_data_file)
        # drop nans
        df = df[df['close'].notna()]

        # Check if Update is neccessary
        last_date = df.iloc[-1]['date']
        last_date = datetime.datetime.strptime(last_date, "%Y-%m-%d")
        final_target_date = datetime.datetime.strptime(self.end_date, "%Y-%m-%d")
        today = datetime.datetime.now().date()
        diff = (final_target_date - last_date).days

        make_update = False

        # Update intrady data if the final date is todayis the current day
        if final_target_date.date() == today and self.interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
            update_df = self.download_data_intraday(last_date, final_target_date, ticker)
            make_update = True

        # Only update daily date if difference is > 4  # Todo change the this to tradeable dates and diff >1
        else:
            if diff > 4:
                logger.info("Doing data update (LastDate: '{}', Target date: '{}')".format(
                    last_date, final_target_date))

                if self.interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
                    update_df = self.download_data_intraday(last_date, final_target_date, ticker)
                else:
                    update_df = YahooDownloader(
                        start_date=last_date.strftime("%Y-%m-%d"),
                        end_date=final_target_date.strftime("%Y-%m-%d"),
                        ticker_list=[ticker],
                        interval=self.interval
                    ).fetch_data()

                make_update = True

            else:
                logger.info("Raw data is recent enough for not doing update".format(raw_data_file))

        if make_update:
            full_df = pd.concat([df, update_df])
            # Drop duplicates
            full_df = full_df.drop_duplicates(subset=['datetime'], keep='first')
            first_date = full_df.date.values[0]
            last_date = full_df.date.values[-1]
            out_name = os.path.join(self.raw_dir, '{}_{}_{}'.format(ticker, first_date, last_date))
            logger.info("Saving new {} data as '{}'".format(self.interval, out_name))
            # Remove old file
            os.remove(raw_data_file)
            full_df.to_pickle(out_name)

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

    def break_up_date_range(self, start, end, intv_size=50):

        if isinstance(start, str):
            start = datetime.datetime.strptime(start, "%Y-%m-%d")
        if isinstance(end, str):
            end = datetime.datetime.strptime(end, "%Y-%m-%d")
        final_end = end
        n_intervals = int(np.ceil((end - start).days/intv_size))
        all_intervals = []
        for i in range(n_intervals):
            if i > 0:
                start += datetime.timedelta(days=intv_size + 1)
            end = start + datetime.timedelta(days=intv_size)
            if end > final_end:

                if start == final_end:
                    # One day is duplicated. Should be removed later
                    start -= datetime.timedelta(days=1)
                    logger.warning("Encountered interval with start == end. Moving start one day back")

                all_intervals.append((start.strftime("%Y-%m-%d"), final_end.strftime("%Y-%m-%d")))
                return all_intervals

            all_intervals.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))



    def download_data_intraday(self, start, end, t):

        today = datetime.datetime.now()
        dr = self.break_up_date_range(start, end)
        full_df = pd.DataFrame()

        # ToDo do this check somewhere else globally for all stocks and nicer
        if isinstance(start, str):
            start = datetime.datetime.strptime(start, "%Y-%m-%d")
        if (today - start).days >= 60:
            logger.warning("Skipping download of some intraday data (ticker: {} start {}) bc older than 60 days".format(
                t, start))

        for start_end in dr:

            if (today - datetime.datetime.strptime(start_end[0], "%Y-%m-%d")).days >= 60:
                continue

            df = YahooDownloader(
                start_date=start_end[0],
                end_date=start_end[1],
                ticker_list=[t],
                interval=self.interval
            ).fetch_data()
            # If using tmp data, it should be checked if this is present to load
            # tmp_out = os.path.join(self.raw_dir, "tmp", '{}_{}_{}'.format(t, start_end[0], start_end[1]))
            # logger.info("Saving new tmp data as '{}'".format(self.raw_dir))
            # df.to_pickle(tmp_out)
            full_df = pd.concat([full_df, df])

        return full_df


    def download_new_ticker_data(self, t):
        logger.info("Downlaoding new Dataset for ticker '{}'".format(self.ticker_list_name))

        if self.interval in ["1m","2m","5m","15m","30m","60m","90m","1h"]:
            df = self.download_data_intraday(self.start_date, self.end_date, t)

        else:

            df = YahooDownloader(
                start_date=self.start_date,
                end_date=self.end_date,
                ticker_list=[t],
                # ticker_list=self.ticker,
                interval=self.interval
            ).fetch_data()

        first_date = df.date.values[0]
        last_date = df.date.values[-1]
        out_name = os.path.join(self.raw_dir, '{}_{}_{}'.format(t, first_date, last_date))
        logger.info("Saving new {} data as '{}'".format(self.interval, out_name))
        df.to_pickle(out_name)

        return df

    def create_dataset(self, preprocess=True):
        # Check if a preprocessed_dataframe_exists
        if os.path.isfile(self.prepro_filename):
            logger.info("Loading preprocessed dataset '{}'".format(self.prepro_filename))
            df = self.load_preprocessed_dataset()
        else:
            logger.info("No preprocessed dataset found. Assembling from Raw data")

            df = pd.DataFrame()
            for ticker in self.ticker:

                # Check if raw data for ticker typpe eixsts and extent if so
                raw_data = self._check_raw_data(ticker)
                if raw_data is not None:
                    ticker_df = self.load_and_update_raw_data(raw_data, ticker)
                else:
                    ticker_df = self.download_new_ticker_data(ticker)

                df = pd.concat([df, ticker_df])
                # df = df.reset_index(drop=True)

            if preprocess:
                logger.info("Preprocessing raw data")
                df = self.preprocess_data(df, config.USE_VIX, config.USE_TURBULENCE)
            logger.info("Saving preprocessed data as '{}'".format(self.prepro_filename))
            df.to_pickle(self.prepro_filename)

        logger.info("Using Ticker: {}".format(self.ticker))
        logger.info("Full dataset shape: {}".format(df.shape))
        df.sort_values(["date", "tic"], ignore_index=True).head()

        return df


    def preprocess_data(self, df, use_vix, use_turbulence):
        # """
        # # Part 4: Preprocess Data
        # Data preprocessing is a crucial step for training a high quality machine learning model. We need to check for missing data and do feature engineering in order to convert the data into a model-ready state.
        # * Add technical indicators. In practical trading, various information needs to be taken into account, for example the historical stock prices, current holding shares, technical indicators, etc. In this article, we demonstrate two trend-following technical indicators: MACD and RSI.
        # * Add turbulence index. Risk-aversion reflects whether an investor will choose to preserve the capital. It also influences one's trading strategy when facing different market volatility level. To control the risk in a worst-case scenario, such as financial crisis of 2007–2008, FinRL employs the financial turbulence index that measures extreme asset price fluctuation.
        # """
        logger.info("Switched off turbulence")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.INDICATORS,
            use_vix=use_vix,
            use_turbulence=use_turbulence,
            user_defined_feature=False,
            clean_strategy=config.CLEAN_STRATEGY
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


