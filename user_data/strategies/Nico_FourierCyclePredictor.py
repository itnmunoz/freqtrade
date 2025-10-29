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

    '''
    def fourier_predict(self, signal: np.ndarray, n_freqs: int = 5):
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        idx = np.argsort(np.abs(fft))[-n_freqs:]
        fft_filtered = np.zeros_like(fft)
        fft_filtered[idx] = fft[idx]
        reconstructed = np.fft.ifft(fft_filtered).real
        t_next = len(signal)
        prediction = sum(
            np.abs(fft[i]) * np.cos(2 * np.pi * freqs[i] * t_next + np.angle(fft[i]))
            for i in idx
        )
        return reconstructed, prediction
    '''

    def fourier_predict(self, signal: np.ndarray, n_freqs: int = 5):
        # Transformada directa sin centrar
        fft = np.fft.fft(signal)

        # Filtrar altas frecuencias, conservando la componente DC
        fft[n_freqs:-n_freqs] = 0

        # Reconstrucción sin sumar la media manualmente
        reconstructed = np.fft.ifft(fft).real

        # Predicción: extrapolar una vela más
        t_pred = len(signal)
        freqs = np.fft.fftfreq(len(signal))
        pred = np.sum([
            2 * np.abs(fft[k]) * np.cos(2 * np.pi * freqs[k] * t_pred + np.angle(fft[k]))
            for k in range(1, n_freqs + 1)
        ]) / len(signal)

        return reconstructed, pred

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        window = 256
        n_freqs = 5

        df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)
        df['fourier_pred'] = np.nan
        df['reconstructed'] = np.nan
        df['cycle_slope'] = np.nan
        df['cycle_min'] = np.nan

        for i in range(window, len(df)):
            signal = df['cycle'].values[i - window:i]
            reconstructed, pred = self.fourier_predict(signal, n_freqs=n_freqs)

            # Asignar solo el último valor de cada cálculo a la vela actual
            df.loc[df.index[i], 'reconstructed'] = reconstructed[-1]
            df.loc[df.index[i], 'cycle_slope'] = np.gradient(reconstructed)[-1]

            # Detectar si la última vela es mínimo local
            min_idx = argrelextrema(reconstructed, np.less_equal, order=5)[0]
            if window - 1 in min_idx:
                df.loc[df.index[i], 'cycle_min'] = reconstructed[-1]

            # Predicción Fourier
            df.loc[df.index[i], 'fourier_pred'] = pred

        return df

    '''
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
    '''

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
