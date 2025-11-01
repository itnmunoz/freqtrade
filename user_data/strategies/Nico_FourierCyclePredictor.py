from freqtrade.strategy import IStrategy
from pandas import DataFrame
from scipy.signal import savgol_filter, argrelextrema
import numpy as np


class FourierCycleInflection(IStrategy):
    timeframe = '1m'

    minimal_roi = {
        "0": 0.05,    # 5% en cualquier momento
        "60": 0.02,   # 3% después de 60 minutos
        "120": 0.01,  # 1% después de 120 minutos
        "180": 0
    }

    stoploss = -0.015
    use_custom_exit_trend = True  # Evitar salir tarde

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
        )/t_next

        return reconstructed, prediction

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)
        df['cycle_slope'] = np.nan
        df['cycle_min'] = np.nan
        df['fourier_pred'] = np.nan
        df['reconstructed'] = np.nan

        if len(df) >= 256:
            signal = df['cycle'].values[-256:]

            '''
            reconstructed, pred = self.fourier_predict(signal)
            df.loc[df.index[-256:], 'reconstructed'] = reconstructed

            # Detectar mínimos locales en la señal reconstruida
            min_idx = argrelextrema(reconstructed, np.less_equal, order=5)[0]
            df.loc[df.index[-256:], 'cycle_min'] = np.nan
            df.loc[df.index[-256:][min_idx],'cycle_min'] = reconstructed[min_idx]

            # Guardar predicción
            df.loc[df.index[-1], 'fourier_pred'] = pred
            '''

            # Calcular pendiente sobre la señal original suavizada
            slope = np.gradient(df['cycle'].values[-256:])
            smoothed_slope = self.lowpass_filter(
                slope, kernel_size=7, window='hamming')
            df.loc[df.index[-256:], 'cycle_slope'] = smoothed_slope

            # Superponer en la gráica en la misma ordenada
            df['cycle_slope_offset'] = df['cycle_slope']*10 + np.mean(signal)
            df['cycle_slope_mean'] = np.mean(signal)

        return df

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        '''
        df.loc[
            (
                (df['cycle_slope'] > 0) &
                (df['cycle_slope'].shift(1) <= 0)
            ),
            'buy'
        ] = 1
        return df
        '''
        # Eliminar índices duplicados para evitar errores de reindexado
        df = df[~df.index.duplicated(keep='last')]

        # Inicializar columnas si no existen
        df['buy'] = 0
        df['buy_tag'] = ''

        # Condición de entrada: cruce de pendiente negativa a positiva
        buy_condition = (df['cycle_slope'] > 0) & (
            df['cycle_slope'].shift(1) <= 0)

        # Asignar señal y etiqueta
        df.loc[buy_condition, 'buy'] = 1
        df.loc[buy_condition, 'buy_tag'] = 'slope_up'

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['cycle_slope'] < 0)  # tramo descendente
                # (df['fourier_pred'] < df['close']) &
            ),
            'sell'
        ] = 1
        return df

    @staticmethod
    def lowpass_filter(signal: np.ndarray, kernel_size: int = 5, window: str = 'hamming') -> np.ndarray:
        """
        Aplica un filtro paso bajo por convolución a una señal 1D.

        Parameters:
        - signal: np.ndarray. Señal original.
        - kernel_size: int. Tamaño del kernel de suavizado.
        - window: str. Tipo de ventana ('hamming', 'rect', 'gaussian').

        Returns:
        - np.ndarray. Señal suavizada.
        """
        if window == 'hamming':
            kernel = np.hamming(kernel_size)
        elif window == 'gaussian':
            from scipy.signal.windows import gaussian
            kernel = gaussian(kernel_size, std=kernel_size / 6)
        elif window == 'rect':
            kernel = np.ones(kernel_size)
        else:
            raise ValueError(f"Ventana no soportada: {window}")

        kernel /= kernel.sum()
        smoothed = np.convolve(signal, kernel, mode='same')
        return smoothed

    '''
    def fourier_predict(self, signal: np.ndarray, n_freqs: int = 5):
        mean = np.mean(signal)
        centered = signal - mean

        fft = np.fft.fft(centered)
        fft[n_freqs:-n_freqs] = 0

        reconstructed = np.fft.ifft(fft).real + mean

        t_pred = len(centered)
        freqs = np.fft.fftfreq(len(centered))

        # Predicción sin factor 2 ni normalización artificial
        pred = sum(
            np.abs(fft[k]) * np.cos(2 * np.pi * freqs[k] * t_pred + np.angle(fft[k]))
            for k in range(1, n_freqs + 1)
        )/t_pred + mean

        return reconstructed, pred
    '''
    '''
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
