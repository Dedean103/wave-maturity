import pandas as pd
import numpy as np

def is_downtrend_five_wave_strict(points_prices):
    """
    严格判断是否为下跌五浪结构
    要求：P1>P2<P3>P4<P5>P6，且每个点都严格满足条件
    """
    if len(points_prices) != 6:
        return False
    
    # 检查交替性：P1>P2, P2<P3, P3>P4, P4<P5, P5>P6
    for i in range(5):
        if i % 2 == 0:  # 偶数位置应该是高点
            if points_prices.iloc[i] <= points_prices.iloc[i+1]:
                return False
        else:  # 奇数位置应该是低点
            if points_prices.iloc[i] >= points_prices.iloc[i+1]:
                return False
    
    return True

def is_downtrend_five_wave_relaxed(points_prices):
    """
    宽松判断是否为下跌五浪结构
    允许轻微的偏差，主要用于识别近似五浪结构
    """
    if len(points_prices) != 6:
        return False
    
    # 检查主要趋势：P1>P3>P5 和 P2<P4<P6
    if not (points_prices.iloc[0] > points_prices.iloc[2] > points_prices.iloc[4]):
        return False
    
    if not (points_prices.iloc[1] < points_prices.iloc[3] < points_prices.iloc[5]):
        return False
    
    return True

def find_all_wave_patterns(close_prices, extremas):
    """
    从极值点中识别所有可能的波浪模式
    
    参数:
        close_prices (pd.Series): 收盘价序列
        extremas (list): 极值点日期列表
    
    返回:
        tuple: (严格波浪列表, 宽松波浪列表)
    """
    strict_waves = []
    relaxed_waves = []
    extremas_indices = [close_prices.index.get_loc(idx) for idx in extremas]
    
    for i in range(len(extremas_indices) - 5):
        points_indices = extremas_indices[i:i+6]
        points_prices = close_prices.iloc[points_indices]
        
        is_alternating = True
        for j in range(len(points_prices) - 1):
            # 判断逻辑符合下跌五浪 P1>P2<P3>P4<P5>P6
            if j % 2 == 0 and points_prices.iloc[j] <= points_prices.iloc[j+1]:
                is_alternating = False
                break
            elif j % 2 != 0 and points_prices.iloc[j] >= points_prices.iloc[j+1]:
                is_alternating = False
                break
        
        if is_alternating:
            if is_downtrend_five_wave_strict(points_prices): 
                strict_waves.append(points_indices)
            elif is_downtrend_five_wave_relaxed(points_prices): 
                relaxed_waves.append(points_indices)
            
    return strict_waves, relaxed_waves

def remove_overlapping_waves(waves_list, close_prices):
    """
    移除重叠的波浪，保留不重叠的波浪
    
    参数:
        waves_list (list): 波浪索引列表
        close_prices (pd.Series): 收盘价序列
    
    返回:
        list: 不重叠的波浪列表
    """
    if not waves_list: 
        return []
    
    waves_with_dates = []
    for indices in waves_list:
        start_date = close_prices.index[indices[0]]
        end_date = close_prices.index[indices[-1]]
        waves_with_dates.append({'start': start_date, 'end': end_date, 'indices': indices})
    
    waves_with_dates.sort(key=lambda x: x['start'])
    non_overlapping_waves = []
    
    for wave in waves_with_dates:
        is_overlapping = False
        for accepted_wave in non_overlapping_waves:
            if wave['start'] < accepted_wave['end'] and accepted_wave['start'] < wave['end']:
                is_overlapping = True
                break
        if not is_overlapping:
            non_overlapping_waves.append(wave)
    
    return [w['indices'] for w in non_overlapping_waves]
