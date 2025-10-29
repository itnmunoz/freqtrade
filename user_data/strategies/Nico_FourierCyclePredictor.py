from freqtrade.strategy import IStrategy
from pandas import DataFrame
from scipy.signal import savgol_filter, argrelextrema
import numpy as np


class FourierCycleInflection(IStrategy):
    timeframe = '5m'
    startup_candle_count = 300

    minimal_roi = {"0": 0.02, "30": 0.01, "60": 0}
    stoploss = -0.015
    use_custom_stoploss = False

    def fourier_predict(self, signal: np.ndarray, n_freqs: int = 5):
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        idx = np.argsort(np.abs(fft))[-n_freqs:]
        fft_filtered = np.zeros_like(fft)
        fft_filtered[idx] = fft[idx]
        reconstructed = np.fft.ifft(fft_filtered).real
        t_next = len(signal)
        prediction = sum(
            np.abs(fft[i]) * np.cos(2 * np.pi *
                                    freqs[i] * t_next + np.angle(fft[i]))
            for i in idx
        )
        return reconstructed, prediction

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)
        df['fourier_pred'] = np.nan
        df['reconstructed'] = np.nan
        df['cycle_slope'] = np.nan
        df['cycle_min'] = np.nan

        if len(df) >= 256:
            signal = df['cycle'].values[-256:]
            reconstructed, pred = self.fourier_predict(signal)
            df.loc[df.index[-256:], 'reconstructed'] = reconstructed

            # Detectar mínimos locales en la señal reconstruida
            min_idx = argrelextrema(reconstructed, np.less_equal, order=5)[0]
            df.loc[df.index[-256:], 'cycle_min'] = np.nan
            df.loc[df.index[-256:][min_idx],
                   'cycle_min'] = reconstructed[min_idx]

            # Calcular pendiente de la señal reconstruida
            slope = np.gradient(reconstructed)
            df.loc[df.index[-256:], 'cycle_slope'] = slope

            # Guardar predicción
            df.loc[df.index[-1], 'fourier_pred'] = pred

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['fourier_pred'] > df['close']) &
                (df['cycle_slope'] > 0) &  # tramo creciente
                # venimos de un mínimo reciente
                (df['cycle_min'].notnull().shift(1))
            ),
            'buy'
        ] = 1
        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['fourier_pred'] < df['close']) &
                (df['cycle_slope'] < 0)  # tramo descendente
            ),
            'sell'
        ] = 1
        return df
