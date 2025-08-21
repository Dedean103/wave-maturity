import pandas as pd
import numpy as np

def is_downtrend_five_wave_strict(prices):
    """
    严格判断是否为下跌五浪结构
    要求：P1>P2<P3>P4<P5>P6，且每个点都严格满足条件
    """
    if len(prices) != 6:
        return False
    
    p0, p1, p2, p3, p4, p5 = prices.values
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5):
        return False
    if p2 >= p0 or p4 >= p2 or p3 >= p1 or p5 >= p3 or p4 <= p1:
        return False
    return True

def is_downtrend_five_wave_relaxed(prices):
    """
    宽松判断是否为下跌五浪结构
    允许轻微的偏差，主要用于识别近似五浪结构
    """
    if len(prices) != 6:
        return False
    
    p0, p1, p2, p3, p4, p5 = prices.values
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5):
        return False
    if p2 >= p0 or p4 >= p0 or p5 >= p1 or p4 <= p1:
        return False
    return True

def is_downtrend_seven_wave(prices):
    """
    检查一个8个点的波浪是否符合下跌7浪结构，并加入P8 < P6的条件
    """
    if len(prices) != 8:
        return False
    
    p0, p1, p2, p3, p4, p5, p6, p7 = prices.values
    
    # 检查交替高低点
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5 and p5 < p6 and p6 > p7):
        return False
        
    # 添加新条件: P8 必须低于 P6
    if not (p7 < p5):
        return False
    
    return True

def find_downtrend_wave_patterns(close_prices, extremas, length=6):
    """
    泛化波浪寻找函数，可以寻找指定长度的波浪
    
    参数:
        close_prices (pd.Series): 收盘价序列
        extremas (list): 极值点日期列表
        length (int): 波浪长度（6点或8点）
    
    返回:
        list: 波浪字典列表，包含indices和type
    """
    waves = []
    extremas_indices = [close_prices.index.get_loc(idx) for idx in extremas]
    
    for i in range(len(extremas_indices) - length + 1):
        points_indices = extremas_indices[i:i+length]
        points_prices = close_prices.iloc[points_indices]
        
        # 检查是否为交替的峰谷
        is_alternating = True
        for j in range(len(points_prices) - 1):
            if j % 2 == 0 and points_prices.iloc[j] <= points_prices.iloc[j+1]:
                is_alternating = False
                break
            elif j % 2 != 0 and points_prices.iloc[j] >= points_prices.iloc[j+1]:
                is_alternating = False
                break
        
        if is_alternating:
            if length == 6:
                if is_downtrend_five_wave_strict(points_prices): 
                    waves.append({'indices': points_indices, 'type': 'strict'})
                elif is_downtrend_five_wave_relaxed(points_prices): 
                    waves.append({'indices': points_indices, 'type': 'relaxed'})
            elif length == 8:
                if is_downtrend_seven_wave(points_prices): 
                    waves.append({'indices': points_indices, 'type': 'merged'})
            
    return waves

def handle_overlapping_waves(all_waves, close_prices, all_extremas):
    """
    处理重叠波浪的高级函数，将重叠的波浪合并为7浪结构
    
    参数:
        all_waves (list): 所有检测到的波浪
        close_prices (pd.Series): 收盘价序列
        all_extremas (list): 所有极值点
    
    返回:
        tuple: (最终波浪列表, 合并波浪列表)
    """
    if not all_waves:
        return [], []

    waves_to_process = all_waves[:]
    final_waves = []
    merged_waves = []
    i = 0
    
    while i < len(waves_to_process):
        current_wave = waves_to_process[i]
        
        overlap_found = False
        j = i + 1
        while j < len(waves_to_process):
            next_wave = waves_to_process[j]
            
            # Check for overlap
            if next_wave['indices'][0] <= current_wave['indices'][-1]:
                overlap_found = True
                print(f"检测到波浪重叠：波浪 A ({close_prices.index[current_wave['indices'][0]].date()} - {close_prices.index[current_wave['indices'][-1]].date()}) 与 波浪 B ({close_prices.index[next_wave['indices'][0]].date()} - {close_prices.index[next_wave['indices'][-1]].date()})")
                
                # 定义新的搜索范围
                start_index = current_wave['indices'][0]
                end_index = next_wave['indices'][-1]
                
                # 寻找该范围内的所有极值点
                search_extremas = [e for e in all_extremas if close_prices.index.get_loc(e) >= start_index and close_prices.index.get_loc(e) <= end_index]
                
                # 在新的范围和极值点中寻找一个8点的波浪
                new_wave = find_downtrend_wave_patterns(close_prices, search_extremas, length=8)
                
                if new_wave:
                    merged_waves.append(new_wave[0])
                    print(f" -> 成功找到新的合并波浪（8点）: 从 {close_prices.index[new_wave[0]['indices'][0]].date()} 到 {close_prices.index[new_wave[0]['indices'][-1]].date()}")
                else:
                    print(" -> 未能在重叠区域内找到新的8点波浪，两个波浪均被移除。")

                # 移除重叠的两个波浪，并跳过下一个波浪
                i += 2
                break
            else:
                j += 1
        
        if not overlap_found:
            final_waves.append(current_wave)
            i += 1
            
    return final_waves, merged_waves

# Legacy functions for backward compatibility
def find_all_wave_patterns(close_prices, extremas):
    """
    Legacy function - 向后兼容
    """
    waves = find_downtrend_wave_patterns(close_prices, extremas, length=6)
    strict_waves = [w['indices'] for w in waves if w['type'] == 'strict']
    relaxed_waves = [w['indices'] for w in waves if w['type'] == 'relaxed']
    return strict_waves, relaxed_waves

def remove_overlapping_waves(waves_list, close_prices):
    """
    Legacy function - 向后兼容
    移除重叠的波浪，保留不重叠的波浪
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