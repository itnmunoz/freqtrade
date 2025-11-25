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
        "0": 0.04,
        "30": 0.03,
        "60": 0.02,
        "120": 0.01,
        "180": 0
    }

    stoploss = -0.005
    use_custom_exit_trend = True  # Activado por utilizar custom_exit()

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

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:

        # df['cycle'] = savgol_filter(df['close'], window_length=21, polyorder=3)
        df['cycle'] = ta.EMA(df['close'], timeperiod=7)

        df['cycle_slope'] = np.nan
        df['cycle_min'] = np.nan
        df['fourier_pred'] = np.nan
        df['reconstructed'] = np.nan

        # if len(df) >= 256:
        signal = df['cycle'].values[-256:]
        mean = np.mean(signal)

        # Calcular pendiente sobre la señal original suavizada
        slope = np.gradient(df['cycle'].values[-256:])
        # smoothed_slope = self.lowpass_filter(slope, kernel_size=7, window='hamming')
        df.loc[df.index[-256:], 'cycle_slope'] = slope  # smoothed_slope

        # Superponer en la gráica en la misma ordenada
        df['cycle_slope_offset'] = df['cycle_slope']*10 + mean
        df['cycle_slope_mean'] = mean

        # Llamar a la función de anticipación
        df = self.anticipate_inflection(
            df, slope_col='cycle_slope', lookback=3, horizon=(3, 4))
        df['y0_proj_offset'] = df['y0_proj']*10 + mean
        df['y1_proj_offset'] = df['y1_proj']*10 + mean

        # Demasiadas oscilaciones
        df = self.detect_cycle_frequency(df, slope_col='cycle_slope', window=10, max_turns=4)

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas si no existen
        # df['enter_long'] = 0
        # df['enter_tag'] = ''

        # Con datos actuales: cruce de pendiente negativa a positiva y tendencia creciente en derivada 4 muestras seguidas
        prediction_0 = (
            (df['cycle_slope'] > 0) &
            (df['cycle_slope'].shift(1) <= 0) &
            (df['cycle_slope'].shift(2) < df['cycle_slope'].shift(1)) &
            (df['cycle_slope'].shift(3) < df['cycle_slope'].shift(2))
        )

        # El predictor a una muestra futura hace inflexión y tendencia creciente en derivada 4 muestras seguidas
        prediction_1 = (
            (df['y1_proj'] > 0) &
            (df['cycle_slope'].shift(1) <= 0) &
            (df['cycle_slope'].shift(2) < df['cycle_slope'].shift(1)) &
            (df['cycle_slope'].shift(3) < df['cycle_slope'].shift(2))
        )

        # Se nos pasa el punto de inflexion, paso por 0 y tendencia 4 velas crecientes
        regression_1 = (
            (df['cycle_slope'] > df['cycle_slope'].shift(1)) &
            (df['cycle_slope'].shift(1) > 0) &
            (df['cycle_slope'].shift(2) <= 0) &
            (df['cycle_slope'].shift(3) < df['cycle_slope'].shift(2))
        )

        # Fuerte tendencia creciente hasta el máximo local
        regression_2 = (
            (df['cycle_slope'].shift(2) > 0) &
            (df['cycle_slope'].shift(1) > 0) &
            (df['cycle_slope'] > 0) &
            (df['cycle_slope'] > df['cycle_slope'].shift(1)) &
            (df['cycle_slope'].shift(1) > df['cycle_slope'].shift(2))
        )

        # BIT OR
        inflection = (prediction_0 | prediction_1 | regression_1)

        df['enter_long'] = inflection.astype(bool)

        # Asignar etiqueta alineada con la señal adelantada
        df.loc[inflection & prediction_0, 'enter_tag'] = 'slope_up'
        df.loc[inflection & prediction_1, 'enter_tag'] = 'entry_anticipation'
        df.loc[inflection & regression_1, 'enter_tag'] = 'regression_up'

        # Resumen
        logger.info(
            f"[ENTRY COUNT] {metadata['pair']} | Total señales: {df['enter_long'].sum()} | Última: {df['enter_long'].iloc[-1]}")

        return df

    '''
    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Inicializar columnas si no existen
        if 'exit_long' not in df.columns:
            df['exit_long'] = False
        if 'exit_tag' not in df.columns:
            df['exit_tag'] = ''

        # Con dactos actuales pendiente negativa y que no se acabe de entrar en la vela anterior
        exit_prediction_0 = (df['cycle_slope'] < 0) & (df['cycle_slope'].shift(1) >= 0)

        # Anticipar la salida si la predicción a una muestra en la derivada prevé inflexión TODO: solo si se va ganando
        exit_prediction_1 = (
            (df['y0_proj'] < 0) &
            (df['y0_proj'].shift(1) >= 0) &
            (df['y0_proj'].shift(2) > df['y0_proj'].shift(1))
        )

        # Calcula la desviación estándar móvil de la pendiente
        df['slope_std'] = df['cycle_slope'].rolling(50).std()
        # Define el umbral dinámico como el negativo de esa desviación
        df['slope_threshold'] = -df['slope_std']
        exit_prediction_2 = (df['cycle_slope'] < df['slope_threshold'])

        # Muestra el último
        logger.info(
            f"[EXIT THRESHOLD] {metadata['pair']} | Último threshold: {df['slope_threshold'].iloc[-1]:.5f} | Derivada actual: {df['cycle_slope'].iloc[-1]:.5f}")

        # Booleans chain y margen 3 velas para evitar salida recién entrado
        exit_condition = (
            (exit_prediction_0 | exit_prediction_2) &
            ~df['enter_long'] &
            ~df['enter_long'].shift(1).astype(bool) &
            ~df['enter_long'].shift(2).astype(bool)
            )

        df.loc[exit_condition, 'exit_long'] = True
        df.loc[exit_prediction_0 & exit_condition, 'exit_tag'] = 'slope_down'
        # df.loc[exit_prediction_1 & exit_condition, 'exit_tag'] = 'exit_anticipation'
        df.loc[exit_prediction_2 & exit_condition, 'exit_tag'] = 'threshold_dynamic'

        # logging
        logger.debug(
            f"[exit] {metadata['pair']} | Señal activa: {df['exit_long'].iloc[-1]}")

        return df
    '''

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # Método requerido por Freqtrade, aunque usemos custom_exit
        df['exit_long'] = False
        df['exit_tag'] = ''
        return df

    # =========================
    # Helpers de condiciones
    # =========================
    def slope_down_condition(self, pair: str, current_time: datetime) -> bool:
        df = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        if 'cycle_slope' not in df.columns:
            return False
        cond = (df['cycle_slope'].iloc[-1] < 0) and (df['cycle_slope'].iloc[-2] >= 0)
        # margen de 3 velas para evitar salida recién entrado
        recent_enter = (
            df['enter_long'].iloc[-1]
            or df['enter_long'].iloc[-2]
            or df['enter_long'].iloc[-3]
        )
        return cond and not recent_enter

    def exit_anticipation_condition(self, pair: str, current_time: datetime) -> bool:
        df = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        if 'cycle_slope' not in df.columns:
            return False
        cond = (
            (df['y0_proj'].iloc[-1] < 0)
            and (df['y0_proj'].iloc[-2] >= 0)
            and (df['y0_proj'].iloc[-3] > df['y0_proj'].iloc[-2])
        )
        recent_enter = (
            df['enter_long'].iloc[-1]
            or df['enter_long'].iloc[-2]
            or df['enter_long'].iloc[-3]
        )
        return cond and not recent_enter

    def threshold_dynamic_condition(self, pair: str, current_time: datetime) -> bool:
        df = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        if 'cycle_slope' not in df.columns:
            return False

        df['slope_std'] = df['cycle_slope'].rolling(50).std()
        df['slope_threshold'] = -df['slope_std']
        cond = df['cycle_slope'].iloc[-1] < df['slope_threshold'].iloc[-1]
        recent_enter = (
            df['enter_long'].iloc[-1]
            or df['enter_long'].iloc[-2]
            or df['enter_long'].iloc[-3]
        )
        return cond and not recent_enter

    # =========================
    # Custom Exit
    # =========================
    def custom_exit(self, pair: str, trade: Trade, current_rate: float,
                    current_time: datetime, **kwargs) -> Optional[str]:

        profit = trade.calc_profit_ratio(current_rate)

        # slope_down → solo si hay beneficio
        if self.slope_down_condition(pair, current_time):
            if profit > self.fee:
                return "slope_down"

        # anticipación → opcional, también solo si hay beneficio
        # if self.exit_anticipation_condition(pair, current_time):
        #     if profit > self.fee:
        #         return "exit_anticipation"

        # threshold_dynamic → siempre válido
        if self.threshold_dynamic_condition(pair, current_time):
            return "threshold_dynamic"

        # stop_loss defensivo
        if profit < self.stoploss:
            return "stop_loss"

        return None

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
