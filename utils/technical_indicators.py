import pandas as pd
import numpy as np

def calculate_rsi(prices, period=14):
    """
    计算相对强弱指数 (RSI)
    
    参数:
        prices (pd.Series): 价格序列
        period (int): RSI 计算周期，默认14
    
    返回:
        pd.Series: RSI 值序列
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi
