from freqtrade.strategy import IStrategy
from pandas import DataFrame
from scipy.signal import savgol_filter, argrelextrema
import numpy as np


class CycleInflectionScalp(IStrategy):
    """
    Estrategia basada en detección de ciclo dominante y puntos de inflexión.
    Usa filtro Savitzky-Golay para suavizar la serie y detectar máximos/mínimos locales.
    """

    timeframe = '5m'
    startup_candle_count = 50

    minimal_roi = {
        "0": 0.015,
        "30": 0.01,
        "60": 0
    }

    stoploss = -0.015
    use_custom_stoploss = False

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Suavizar la serie con filtro Savitzky-Golay
        df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)

        # Detectar mínimos locales
        min_idx = argrelextrema(df['cycle'].values, np.less_equal, order=5)[0]
        df['min'] = np.nan
        df.loc[df.index[min_idx], 'min'] = df['cycle'].iloc[min_idx]

        # Detectar máximos locales
        max_idx = argrelextrema(df['cycle'].values, np.greater_equal, order=5)[0]
        df['max'] = np.nan
        df.loc[df.index[max_idx], 'max'] = df['cycle'].iloc[max_idx]

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['min'].notnull()) &
                (df['close'] > df['cycle'])  # Confirmación de rebote
            ),
            'buy'
        ] = 1
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['max'].notnull()) &
                (df['close'] < df['cycle'])  # Confirmación de caída
            ),
            'sell'
        ] = 1
        return df
