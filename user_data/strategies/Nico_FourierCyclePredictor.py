from freqtrade.strategy import IStrategy
from pandas import DataFrame
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
import talib.abstract as ta

from datetime import datetime
from typing import Optional, Tuple
from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)


class FourierCycleInflection(IStrategy):
    timeframe = '5m'

    minimal_roi = {
        "0": 0.02,
        "60": 0.015,
        "120": 0.01,
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

    @staticmethod
    def anticipate_inflection(df, slope_col='cycle_slope', lookback=3, horizon=(3, 4)):
        # Inicializa columnas si no existen
        for col in ['x0_proj', 'y0_proj', 'x1_proj', 'y1_proj']:
            if col not in df.columns:
                df[col] = np.nan

        for i in range(lookback, len(df)):
            # mean = df.loc[df.index[i], 'cycle_slope_mean']

            y = df[slope_col].iloc[i - lookback + 1:i + 1].values
            x = np.arange(lookback)
            m, b = np.polyfit(x, y, 1)

            if np.isfinite(m) and np.abs(m) > 1e-6 and np.isfinite(b):
                x0, x1 = horizon
                y0 = m * x0 + b
                y1 = m * x1 + b

                # Guardamos x0/y0, x1/y1
                df.loc[df.index[i], 'x0_proj'] = x0
                df.loc[df.index[i], 'y0_proj'] = y0
                df.loc[df.index[i], 'x1_proj'] = x1
                df.loc[df.index[i], 'y1_proj'] = y1

        # Estrategia TODO
        # df.loc[df.index[-1], 'anticipation'] = anticipation
        # df.loc[df.index[-1], 'buy'] = int(anticipation)

        return df

    @staticmethod
    def detect_cycle_frequency(df, slope_col='cycle_slope', window=10, max_turns=2):
        """
        Detecta frecuencia direccional en una serie de pendientes.
        Marca como válidas las zonas donde hay pocos giros de dirección (baja frecuencia).

        Parámetros:
        - df: DataFrame con la serie.
        - slope_col: columna que contiene la pendiente del ciclo.
        - window: tamaño de la ventana móvil.
        - max_turns: número máximo de giros permitidos para considerar el ciclo válido.

        Devuelve:
        - df con columnas 'slope_sign', 'turns', 'valid_cycle'.
        """
        # Signo de la pendiente (+1, -1, 0)
        df['slope_sign'] = np.sign(df[slope_col])

        # Cuenta cambios de signo en la ventana
        df['turns'] = df['slope_sign'].rolling(window=window).apply(
            lambda x: np.count_nonzero(np.diff(x) != 0), raw=True
        )

        # Máscara booleana: zonas con pocos giros
        df['valid_cycle'] = df['turns'] <= max_turns

        return df

    @staticmethod
    def detect_slope_trend(df, slope_col='cycle_slope'):
        """
        Detecta si las últimas 3 velas muestran una pendiente creciente en cycle_slope.
        Marca True en 'cycle_slope_trend' si se cumple la condición.
        """
        df['cycle_slope_trend'] = (
            (df[slope_col].shift(3) < df[slope_col].shift(2)) &
            (df[slope_col].shift(2) < df[slope_col].shift(1)) &
            (df[slope_col].shift(1) < df[slope_col])
        )

        return df

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        # df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)
        df['cycle'] = ta.EMA(df['close'], timeperiod=7)

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
            # smoothed_slope = self.lowpass_filter(slope, kernel_size=7, window='hamming')
            df.loc[df.index[-256:], 'cycle_slope'] = slope  # smoothed_slope

            # Superponer en la gráica en la misma ordenada
            df['cycle_slope_offset'] = df['cycle_slope']*10 + np.mean(signal)
            df['cycle_slope_mean'] = np.mean(signal)

            # Llamar a la función de anticipación
            df = self.anticipate_inflection(
                df, slope_col='cycle_slope_offset', lookback=3, horizon=(3, 4))

            df = self.detect_slope_trend(df, slope_col='cycle_slope')

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas si no existen
        # df['enter_long'] = 0
        # df['enter_tag'] = ''

        # Con datos actuales: cruce de pendiente negativa a positiva y tendencia creciente en derivada 4 muestras seguidas
        prediction_0 = (df['cycle_slope'] > 0) & (
            df['cycle_slope'].shift(1) <= 0) & (df['cycle_slope_trend'])

        # El predictor a una muestra futura hace inflexión y tendencia creciente en derivada 3 muestras seguidas
        prediction_1 = (df['y0_proj'] > 0) & (df['y0_proj'].shift(
            1) <= 0) & (df['y0_proj'].shift(2) < df['y0_proj'].shift(1))

        # Se nos pasa el punto de inflexion, tendencia 3 velas con tendencia todas creciente
        regression_1 = (df['cycle_slope'] > df['cycle_slope'].shift(1)) & (
            df['cycle_slope'].shift(1) > df['cycle_slope'].shift(2)) & (
            (df['cycle_slope'].shift(2) > 0) | (df['cycle_slope'].shift(1) > 0) | (df['cycle_slope'] > 0))

        # Se nos pasa el punto de inflexión y hay fuerte tendencia creciente hasta el máximo local
        regression_2 = (df['cycle_slope'].shift(2) > 0) & (df['cycle_slope'].shift(1) > 0) & (
            df['cycle_slope'] > 0) & (df['cycle_slope'] > df['cycle_slope'].shift(1) | df['cycle_slope'] > df['cycle_slope'].shift(2))

        # BIT OR
        df['inflection'] = prediction_0 | prediction_1 | regression_1 | regression_2

        df['enter_long'] = df['inflection'].astype(int)

        # Asignar etiqueta alineada con la señal adelantada
        df.loc[df['enter_long'], 'enter_tag'] = 'slope_up'

        # Resumen
        logger.info(
            f"[ENTRY COUNT] {metadata['pair']} | Total señales: {df['enter_long'].sum()} | Última: {df['enter_long'].iloc[-1]}")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas si no existen
        # df['exit_long'] = 0
        # df['exit_tag'] = ''

        # Con dactos actuales pendiente negativa
        exit_prediction_0 = (df['cycle_slope'] < 0)

        # Anticipar la salida si la predicción a una muestra en la derivada prevé inflexión
        exit_prediction_1 = (df['y0_proj'] < 0) & (df['y0_proj'].shift(
            1) >= 0) & (df['y0_proj'].shift(2) > df['y0_proj'].shift(1))

        exit_condition = (exit_prediction_0 | exit_prediction_1) & (
            ~df['enter_long'])

        df.loc[exit_condition, 'exit_long'] = 1
        df.loc[exit_prediction_0 & ~df['enter_long'], 'exit_tag'] = 'slope_down'
        df.loc[exit_prediction_1 & ~df['enter_long'], 'exit_tag'] = 'inflection_anticipation'

        # logging
        logger.debug(
            f"[exit] {metadata['pair']} | Señal activa: {df['exit_long'].iloc[-1]}")

        return df

    '''
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                    current_profit: float, **kwargs) -> Optional[Tuple[str, str]]:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1]

        # Verifica si hay señal de salida
        if last_candle.get('exit_long', 0) == 1:
            exit_tag = last_candle.get('exit_tag', 'custom_exit')
            return "sell", exit_tag

        return None
    '''

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
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                current_profit: float, **kwargs) -> Optional[str]:
        df = self.dp.get_pair_dataframe(pair, self.timeframe)

        # Asegura que el DataFrame no esté vacío y tenga la columna
        if df.empty or 'cycle_slope' not in df.columns:
            return None

        # Evalúa la pendiente en la vela actual
        last_slope = df['cycle_slope'].iloc[-1]

        # Salida si la pendiente es negativa
        if last_slope < 0:
            return 'slope_down'

        return None
    '''

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
