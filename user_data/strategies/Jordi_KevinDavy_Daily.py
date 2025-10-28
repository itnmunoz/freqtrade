# --- coding: utf-8 ---
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from freqtrade.persistence import Trade
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class Algo_KevinDavy_Daily(IStrategy):
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
    timeframe = '1d'
    startup_candle_count = 100

    # Gestión de riesgo básica (se puede ajustar)
    # minimal_roi usado para objetivo simple 1:1
    minimal_roi = {"0": 1.0}   # 1:1 -> resultado neto igual al riesgo si se usa como multiplicador (ajústalo si hace falta)
    # Nota: También se puede usar exits con strategy.exit y calcular limit price, pero aquí dejamos ROI simple y custom_stoploss.

    # Se habilita custom stoploss (basado en ATR)
    use_custom_stoploss = True
    # Valor base por si algo falla (seguro)
    stoploss = -0.10

    # Parámetros configurables desde la estrategia
    ema_length = IntParameter(20, 60, default=40, space='buy')   # EMA40 por defecto
    atr_length = IntParameter(10, 30, default=21, space='buy')   # ATR21 por defecto
    atr_multiplier = IntParameter(1, 4, default=2, space='buy')  # multiplicador ATR para SL inicial (2xATR)

    # -----------------------------------------------------
    # Indicadores
    # -----------------------------------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # EMA filter
        df['ema40'] = ta.EMA(df, timeperiod=self.ema_length.value)

        # ATR indicador
        df['atr'] = ta.ATR(df, timeperiod=self.atr_length.value)

        return df

    # -----------------------------------------------------
    # Señales de compra
    # Condiciones (según vídeo / Kevin Davy):
    # 1) high.shift(2) > high.shift(1)
    # 2) low.shift(2) < low.shift(1)
    # 3) close > high.shift(1)
    # 4) close > ema40 (filtro tendencia alcista)
    # -----------------------------------------------------
    def populate_buy_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df.loc[
            (
                (df['high'].shift(2) > df['high'].shift(1)) &    # máximo de hace 2 velas > máximo de hace 1 vela
                (df['low'].shift(2) < df['low'].shift(1)) &      # mínimo de hace 2 velas < mínimo de hace 1 vela
                (df['close'] > df['high'].shift(1)) &            # cierre actual > máximo de hace 1 vela (ruptura)
                (df['close'] > df['ema40'])                      # filtro: precio por encima de EMA40
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
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if 'atr' not in df.columns or df['atr'].isna().all():
                # fallback: no ATR calculado
                logger.debug("custom_stoploss: ATR not available, using default stoploss.")
                return self.stoploss
            atr_now = df['atr'].iloc[-1]
        except Exception as e:
            logger.exception("custom_stoploss: error getting ATR from dp: %s", e)
            return self.stoploss

        # Distancia en precio que supone  multiplier * ATR
        sl_price_distance = atr_now * self.atr_multiplier.value

        # Convertimos a porcentaje relativo respecto al precio actual
        if current_rate <= 0:
            return self.stoploss

        sl_pct = sl_price_distance / current_rate

        # Devolvemos la distancia negativa (freqtrade espera valor negativo)
        return -abs(sl_pct)
