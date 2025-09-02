import pandas as pd
import numpy as np
from utils.technical_indicators import calculate_rsi

def refine_peak_with_price(close_prices, rsi_peak_date, window_days=5):
    """
    Refine RSI-confirmed peak using price data (bidirectional search)
    Look for higher price within window centered around RSI peak
    """
    try:
        peak_idx = close_prices.index.get_loc(rsi_peak_date)
        
        # Define bidirectional search window
        start_idx = max(0, peak_idx - window_days//2)
        end_idx = min(len(close_prices), peak_idx + window_days//2 + 1)
        search_window = close_prices.iloc[start_idx:end_idx]
        
        # Find highest price point in window
        max_price_idx_local = search_window.idxmax()
        max_price = search_window.max()
        
        # Get original RSI peak price for comparison
        original_price = close_prices.iloc[peak_idx]
        
        # If higher price found, adjust peak point
        if max_price > original_price:
            print(f"   -> 价格微调: 峰值从 {rsi_peak_date.date()} (${original_price:.2f}) 调整到 {max_price_idx_local.date()} (${max_price:.2f})")
            return max_price_idx_local
        else:
            return rsi_peak_date
    except:
        return rsi_peak_date

def refine_valley_with_price(close_prices, rsi_valley_date, window_days=5):
    """
    Refine RSI-confirmed valley using price data (bidirectional search)
    Look for lower price within window centered around RSI valley
    """
    try:
        valley_idx = close_prices.index.get_loc(rsi_valley_date)
        
        # Define bidirectional search window
        start_idx = max(0, valley_idx - window_days//2)
        end_idx = min(len(close_prices), valley_idx + window_days//2 + 1)
        search_window = close_prices.iloc[start_idx:end_idx]
        
        # Find lowest price point in window
        min_price_idx_local = search_window.idxmin()
        min_price = search_window.min()
        
        # Get original RSI valley price for comparison
        original_price = close_prices.iloc[valley_idx]
        
        # If lower price found, adjust valley point
        if min_price < original_price:
            print(f"   -> 价格微调: 谷值从 {rsi_valley_date.date()} (${original_price:.2f}) 调整到 {min_price_idx_local.date()} (${min_price:.2f})")
            return min_price_idx_local
        else:
            return rsi_valley_date
    except:
        return rsi_valley_date

def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=15, rsi_rise_ratio=1/3, price_refinement_window=5):
    """
    基于 RSI 变化识别价格极值点
    
    参数:
        close_prices (pd.Series): 收盘价序列
        rsi_period (int): RSI 计算周期
        rsi_drop_threshold (float): RSI 下跌触发阈值
        rsi_rise_ratio (float): RSI 上涨触发比例
        price_refinement_window (int): 价格微调窗口天数
    
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
                # Apply bidirectional price refinement for peak
                refined_peak_date = refine_peak_with_price(close_prices, peak_date, price_refinement_window)
                
                if not extremas or extremas[-1] != refined_peak_date:
                    extremas.append(refined_peak_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'drop', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,  # 保存波峰RSI
                    'valley_rsi': valley_rsi,
                    'change': rsi_drop_change,
                    'threshold': rsi_drop_threshold,
                    'refined_peak_date': refined_peak_date
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
                # Apply bidirectional price refinement for valley
                refined_valley_date = refine_valley_with_price(close_prices, valley_date, price_refinement_window)
                
                if not extremas or extremas[-1] != refined_valley_date:
                    extremas.append(refined_valley_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'rise', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,  # 保存波谷RSI
                    'change': rsi_rise_change,
                    'threshold': rise_threshold,
                    'refined_valley_date': refined_valley_date
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
