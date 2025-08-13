import pandas as pd
import numpy as np
from utils.technical_indicators import calculate_rsi

def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=15, rsi_rise_ratio=1/3):
    """
    基于 RSI 变化识别价格极值点
    
    参数:
        close_prices (pd.Series): 收盘价序列
        rsi_period (int): RSI 计算周期
        rsi_drop_threshold (float): RSI 下跌触发阈值
        rsi_rise_ratio (float): RSI 上涨触发比例
    
    返回:
        tuple: (极值点列表, 触发点列表)
    """
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    extremas = []
    trigger_points = []
    
    print("\n--- 开始 RSI 驱动的峰谷点识别 ---")
    
    if len(rsi_series) < 2: 
        return extremas, trigger_points
    
    # 初始化
    peak_rsi = rsi_series.iloc[0]
    valley_rsi = rsi_series.iloc[0]
    peak_date = rsi_series.index[0]
    valley_date = rsi_series.index[0]
    
    # 追踪上一个完整波段的极值点，用于动态阈值
    last_wave_peak_rsi = peak_rsi
    
    direction = 0  # 0: 初始状态, 1: 上升趋势, -1: 下降趋势
    
    for i in range(1, len(rsi_series)):
        current_rsi = rsi_series.iloc[i]
        current_date = rsi_series.index[i]
        
        if direction >= 0:  # 寻找波峰
            if current_rsi > peak_rsi:
                peak_rsi = current_rsi
                peak_date = current_date
            
            rsi_drop_change = peak_rsi - current_rsi
            if rsi_drop_change >= rsi_drop_threshold:
                if not extremas or extremas[-1] != peak_date:
                    extremas.append(peak_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'drop', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,  # 保存波峰RSI
                    'valley_rsi': valley_rsi,
                    'change': rsi_drop_change,
                    'threshold': rsi_drop_threshold
                })
                
                print(f"[{current_date.date()}] 触发 RSI Drop Trigger!")
                print(f"   -> 确认波峰在: {peak_date.date()}")
                print(f"   -> RSI 从最高点 {peak_rsi:.2f} 跌至 {current_rsi:.2f}，累计下跌 {rsi_drop_change:.2f} 点，满足阈值 {rsi_drop_threshold}。")
                
                # 状态切换：从上升到下降
                direction = -1
                last_wave_peak_rsi = peak_rsi
                valley_rsi = current_rsi
                valley_date = current_date
        
        if direction <= 0:  # 寻找波谷
            if current_rsi < valley_rsi:
                valley_rsi = current_rsi
                valley_date = current_date
            
            rsi_rise_change = current_rsi - valley_rsi
            # 动态阈值计算
            drop_amount_for_threshold = last_wave_peak_rsi - valley_rsi
            rise_threshold = drop_amount_for_threshold * rsi_rise_ratio
            
            if rsi_rise_change >= rise_threshold and drop_amount_for_threshold > 0:
                if not extremas or extremas[-1] != valley_date:
                    extremas.append(valley_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'rise', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,  # 保存波谷RSI
                    'change': rsi_rise_change,
                    'threshold': rise_threshold
                })
                
                print(f"[{current_date.date()}] 触发 RSI Rise Trigger!")
                print(f"   -> 确认波谷在: {valley_date.date()}")
                print(f"   -> RSI 从最低点 {valley_rsi:.2f} 涨至 {current_rsi:.2f}，累计上涨 {rsi_rise_change:.2f} 点，满足阈值 {rise_threshold:.2f}。")
                
                # 状态切换：从下降到上升
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    print("\n--- RSI 驱动的峰谷点识别结束 ---")
    return extremas, trigger_points
