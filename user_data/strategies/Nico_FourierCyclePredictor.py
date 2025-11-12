from freqtrade.strategy import IStrategy
from pandas import DataFrame
from scipy.signal import savgol_filter, argrelextrema
import numpy as np
import talib.abstract as ta

# from datetime import datetime
# from typing import Optional
# from freqtrade.persistence import Trade

import logging
logger = logging.getLogger(__name__)


class FourierCycleInflection(IStrategy):
    timeframe = '5m'

    minimal_roi = {
        "0": 0.05,
        "60": 0.02,
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

    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        '''
        df.loc[
            (
                (df['cycle_slope'] > 0) &
                (df['cycle_slope'].shift(1) <= 0) &
                (df['cycle_slope_trend'])
            ),
            'buy'
        ] = 1
        return df
        '''
        # Eliminar índices duplicados para evitar errores de reindexado
        # df = df[~df.index.duplicated(keep='last')]

        # Inicializar columnas si no existen
        df['buy'] = 0
        df['buy_tag'] = ''

        # Condición de entrada: cruce de pendiente negativa a positiva
        df['inflection'] = (df['cycle_slope'] > 0) & (
            df['cycle_slope'].shift(1) <= 0)
        df['buy'] = df['inflection'].astype(int)

        # Asignar etiqueta alineada con la señal adelantada
        df.loc[df['buy'], 'buy_tag'] = 'slope_up'

        # Logging de las últimas 5 velas
        for i in range(-5, 0):
            fecha = df.index[i]
            slope = df['cycle_slope'].iloc[i]
            inflection = df['inflection'].iloc[i]
            buy = df['buy'].iloc[i]
            tag = df['buy_tag'].iloc[i] if 'buy_tag' in df.columns else None

            logger.info(
                f"[{metadata['pair']}] Vela {i} | Fecha: {fecha} | Slope: {slope:.4f} | Inflection: {inflection} | Buy: {buy} | Tag: {tag}"
            )

        # Resumen
        logger.info(
            f"[BUY COUNT] {metadata['pair']} | Total señales: {df['buy'].sum()} | Última: {df['buy'].iloc[-1]}")

        return df

    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        '''
        df.loc[
            (
                (df['cycle_slope'] < 0)  # tramo descendente
                # (df['fourier_pred'] < df['close']) &
            ),
            'sell'
        ] = 1
        '''
        # Inicializar columnas si no existen
        df['sell'] = 0
        df['sell_tag'] = ''

        sell_condition = df['cycle_slope'] < 0

        df.loc[sell_condition, 'sell'] = 1
        df.loc[sell_condition, 'sell_tag'] = 'slope_down'

        # logging
        logger.debug(
            f"[SELL] {metadata['pair']} | Señal activa: {df['sell'].iloc[-1]}")

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
