import pandas as pd
import numpy as np

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
            print(f"价格微调: P0从 {rsi_peak_date.date()} ({original_price:.2f}) 调整到 {max_price_idx_local.date()} ({max_price:.2f})")
            return max_price_idx_local
        else:
            return rsi_peak_date
    except:
        return rsi_peak_date

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

def find_continuous_waves_from_recent_highs(close_prices, recent_days=50, rsi_period=14, 
                                          rsi_drop_threshold=10, rsi_rise_ratio=1/3, 
                                          price_refinement_window=5):
    """
    基于最近高RSI点的连续波浪检测新逻辑
    
    参数:
        close_prices (pd.Series): 收盘价序列
        recent_days (int): 查找P0的回望天数
        rsi_period (int): RSI计算周期
        rsi_drop_threshold (int): RSI下跌阈值
        rsi_rise_ratio (float): RSI上涨比例
        price_refinement_window (int): 价格微调窗口
    
    返回:
        tuple: (所有波浪列表, 所有触发点列表)
    """
    from utils.technical_indicators import calculate_rsi
    from core.rsi_analysis import find_extremas_with_rsi
    
    print(f"\n=== 开始基于高RSI点的连续波浪检测 ===")
    print(f"参数: 回望天数={recent_days}, RSI周期={rsi_period}, 下跌阈值={rsi_drop_threshold}, 上涨比例={rsi_rise_ratio}")
    
    # 计算完整数据的RSI
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    
    all_waves = []
    all_trigger_points = []
    
    # 从RSI数据开始位置开始搜索（跳过NaN值）
    rsi_start_idx = close_prices.index.get_loc(rsi_series.index[0])
    used_data_end = rsi_start_idx  # 跟踪已使用数据的结束位置
    
    wave_count = 0
    max_iterations = 50  # 防止无限循环
    
    while used_data_end < len(close_prices) - recent_days and wave_count < max_iterations:
        wave_count += 1
        print(f"\n--- 搜索第 {wave_count} 个波浪 ---")
        
        # 定义搜索窗口
        search_start = used_data_end
        search_end = min(search_start + recent_days, len(close_prices))
        
        if search_end - search_start < recent_days:
            print(f"剩余数据不足 {recent_days} 天，停止搜索")
            break
            
        # 获取搜索窗口内的数据
        window_close = close_prices.iloc[search_start:search_end]
        
        # 只获取RSI数据中存在的日期
        window_dates = window_close.index
        available_rsi_dates = rsi_series.index.intersection(window_dates)
        
        if len(available_rsi_dates) == 0:
            print(f"窗口内没有可用的RSI数据，停止搜索")
            break
            
        window_rsi = rsi_series[available_rsi_dates]
            
        # 找到最高RSI点作为P0候选
        max_rsi_idx = window_rsi.idxmax()
        max_rsi_value = window_rsi.loc[max_rsi_idx]
        
        print(f"搜索窗口: {available_rsi_dates[0].date()} 到 {available_rsi_dates[-1].date()} (共{len(available_rsi_dates)}天)")
        print(f"最高RSI点: {max_rsi_idx.date()}, RSI值: {max_rsi_value:.2f}")
        
        # 价格微调：在微调窗口内寻找最高价格点
        refinement_start = max(0, close_prices.index.get_loc(max_rsi_idx) - price_refinement_window//2)
        refinement_end = min(len(close_prices), close_prices.index.get_loc(max_rsi_idx) + price_refinement_window//2 + 1)
        
        refinement_window = close_prices.iloc[refinement_start:refinement_end]
        if len(refinement_window) > 1:
            max_price_idx = refinement_window.idxmax()
            if max_price_idx != max_rsi_idx:
                print(f"价格微调: P0从 {max_rsi_idx.date()} ({close_prices[max_rsi_idx]:.2f}) 调整到 {max_price_idx.date()} ({close_prices[max_price_idx]:.2f})")
                p0_date = max_price_idx
            else:
                p0_date = max_rsi_idx
        else:
            p0_date = max_rsi_idx
            
        # 从P0开始，寻找完整数据中的后续极值点
        p0_index = close_prices.index.get_loc(p0_date)
        remaining_data = close_prices.iloc[p0_index:]
        
        print(f"确定P0: {p0_date.date()}, 开始从此点寻找后续波浪...")
        
        # 使用现有的RSI驱动极值检测，但只处理P0之后的数据
        extremas, trigger_points = find_extremas_with_rsi(
            remaining_data, 
            rsi_period=rsi_period,
            rsi_drop_threshold=rsi_drop_threshold,
            rsi_rise_ratio=rsi_rise_ratio,
            price_refinement_window=price_refinement_window
        )
        
        # 将P0添加到极值点列表开头（如果不在其中）
        if not extremas or extremas[0] != p0_date:
            extremas.insert(0, p0_date)
            
        print(f"在P0后发现 {len(extremas)} 个极值点")
        
        # 寻找6点波浪
        wave_patterns_6 = find_downtrend_wave_patterns(close_prices, extremas, length=6)
        
        # 寻找8点波浪  
        wave_patterns_8 = find_downtrend_wave_patterns(close_prices, extremas, length=8)
        
        # 合并所有发现的波浪
        current_waves = wave_patterns_6 + wave_patterns_8
        
        if current_waves:
            print(f"发现 {len(current_waves)} 个波浪结构")
            all_waves.extend(current_waves)
            all_trigger_points.extend(trigger_points)
            
            # 更新已使用数据的结束位置到最后一个波浪的结束点
            latest_end = max([close_prices.index.get_loc(close_prices.index[w['indices'][-1]]) for w in current_waves])
            used_data_end = latest_end + 1
            
            print(f"更新已使用数据位置到索引 {used_data_end}")
        else:
            print("未发现波浪结构，向前移动搜索窗口")
            used_data_end = search_start + recent_days // 2  # 移动半个窗口
    
    print(f"\n=== 连续波浪检测完成 ===")
    print(f"总共发现 {len(all_waves)} 个波浪结构")
    
    return all_waves, all_trigger_points