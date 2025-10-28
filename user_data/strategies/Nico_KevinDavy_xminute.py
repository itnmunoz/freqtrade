# --- coding: utf-8 ---
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from freqtrade.persistence import Trade
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Nico_KevinDavy_xminute(IStrategy):
    """
    Estrategia 3 del vídeo adaptada a Freqtrade:
    - Patrón de velas (condiciones de máximos/mínimos y cierre rompedor)
    - Filtro de tendencia: EMA 40
    - Stop Loss inicial = 2 * ATR(21)
    - Take Profit = R:R 1:1 (se gestiona con minimal_roi o cierre manual)
    - Timeframe por defecto: '1d' (seguir el vídeo). Cambiar si quieres otra temporalidad.
    """

    # -----------------------------------------------------
    # Configuración básica
    # -----------------------------------------------------
    timeframe = '5m'

    startup_candle_count = 50

    # Gestión de riesgo básica (se puede ajustar)
    minimal_roi = {
        "0": 0.015,   # 15% si se alcanza rápido
        "30": 0.01,   # después de 30 minutos, acepta 1%
        "60": 0       # después de 1 hora, acepta 0%
    }

    # Se habilita custom stoploss (basado en ATR)
    use_custom_stoploss = True

    # Valor base por si algo falla (seguro)
    stoploss = -0.015

    # Parámetros configurables desde la estrategia adaptados a 15 minutos
    ema_length = IntParameter(5, 15, default=10, space='buy')
    atr_length = IntParameter(5, 14, default=7, space='buy')
    atr_multiplier = DecimalParameter(1.0, 2.0, default=1.2, space='buy')

    # -----------------------------------------------------
    # Indicadores
    # -----------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['ema'] = ta.EMA(df, timeperiod=self.ema_length.value)  # EMA filter
        df['atr'] = ta.ATR(df, timeperiod=self.atr_length.value)  # ATR indicador
        return df

    # -----------------------------------------------------
    # Señales de compra scalping
    # -----------------------------------------------------
    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['close'] > df['high'].shift(1)) &
                (df['close'] > df['ema'])
            ),
            'buy'
        ] = 1
        return df

    # -----------------------------------------------------
    # Señales de venta (para posiciones SHORT si quieres)
    # Condiciones simétricas:
    # 1) high.shift(2) < high.shift(1)
    # 2) low.shift(2) > low.shift(1)
    # 3) close < low.shift(1)
    # 4) close < ema40 (filtro bajista)
    # -----------------------------------------------------
    '''
    def populate_sell_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['high'].shift(2) < df['high'].shift(1)) &
                (df['low'].shift(2) > df['low'].shift(1)) &
                (df['close'] < df['low'].shift(1)) &
                (df['close'] < df['ema40'])
            ),
            'sell'
        ] = 1
        return df
    '''
    # -----------------------------------------------------
    # custom_stoploss: calcula SL dinámico basado en ATR del timeframe configurado.
    # - SL inicial = atr_multiplier * ATR
    # - Se devuelve un valor negativo que representa la distancia relativa (porcentaje)
    # -----------------------------------------------------
    def custom_stoploss(self,
                        pair: str,
                        trade: Trade,
                        current_time,
                        current_rate: float,
                        current_profit: float,
                        **kwargs) -> Optional[float]:

        # Intentamos obtener el ATR más reciente del data provider
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)

        if 'atr' not in df.columns or df['atr'].isna().all():
            return self.stoploss
        
        atr_now = df['atr'].iloc[-1]

        sl_price_distance = atr_now * self.atr_multiplier.value

        if current_rate <= 0:
            return self.stoploss
        
        sl_pct = sl_price_distance / current_rate

        return -abs(sl_pct)

