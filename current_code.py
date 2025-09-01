import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# All wave identification and plotting functions (Current Implementation)
# =========================================================

def is_downtrend_five_wave_strict(prices):
    if len(prices) != 6: return False
    p0, p1, p2, p3, p4, p5 = prices.values
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5): return False
    if p2 >= p0 or p4 >= p2 or p3 >= p1 or p5 >= p3 or p4 <= p1: return False
    return True

def is_downtrend_five_wave_relaxed(prices):
    if len(prices) != 6: return False
    p0, p1, p2, p3, p4, p5 = prices.values
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5): return False
    if p2 >= p0 or p4 >= p0 or p5 >= p1 or p4 <= p1: return False
    return True

def is_downtrend_seven_wave(prices):
    """
    Check if an 8-point wave follows downtrend 7-wave structure with P8 < P6 condition
    """
    if len(prices) != 8:
        return False
    p0, p1, p2, p3, p4, p5, p6, p7 = prices.values
    # Check alternating high-low points
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5 and p5 < p6 and p6 > p7):
        return False
        
    # Add new condition: P8 must be lower than P6
    if not (p7 < p5):
        return False
    
    return True

def calculate_rsi(series, period=14):
    """Calculate RSI using exponential weighted moving average"""
    delta = series.diff().dropna()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(com=period - 1, adjust=False).mean()
    avg_loss = losses.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def find_downtrend_wave_patterns(close_prices, extremas, length=6, trend_threshold=0.95):
    """
    Generalized wave finding function for specified length waves with overall trend validation
    """
    waves = []
    # Sort extremas chronologically to ensure proper temporal order
    extremas_sorted = sorted(extremas)
    extremas_indices = [close_prices.index.get_loc(idx) for idx in extremas_sorted]
    
    for i in range(len(extremas_indices) - length + 1):
        points_indices = extremas_indices[i:i+length]
        points_prices = close_prices.iloc[points_indices]
        
        # First check pattern validation
        pattern_valid = False
        wave_type = None
        
        if length == 6:
            if is_downtrend_five_wave_strict(points_prices): 
                pattern_valid = True
                wave_type = 'strict'
            elif is_downtrend_five_wave_relaxed(points_prices): 
                pattern_valid = True
                wave_type = 'relaxed'
        elif length == 8:
            if is_downtrend_seven_wave(points_prices): 
                pattern_valid = True
                wave_type = 'merged'
        
        # If pattern is valid, check overall trend direction
        if pattern_valid:
            is_trend_valid, trend_ratio, trend_info = validate_wave_overall_trend(
                close_prices, points_indices, trend_threshold
            )
            
            if is_trend_valid:
                waves.append({
                    'indices': points_indices, 
                    'type': wave_type,
                    'trend_ratio': trend_ratio,
                    'trend_info': trend_info
                })
                print(f"✓ Valid {wave_type} wave: {trend_info['start_date']} to {trend_info['end_date']}, trend ratio: {trend_ratio:.3f}")
            else:
                print(f"✗ Rejected {wave_type} wave (uptrend): {trend_info['start_date']} to {trend_info['end_date']}, trend ratio: {trend_ratio:.3f}")
            
    return waves

def handle_overlapping_waves(all_waves, close_prices, all_extremas):
    """
    Handle overlapping waves by attempting to merge them into 7-wave patterns
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
                
                # Define new search range
                start_index = current_wave['indices'][0]
                end_index = next_wave['indices'][-1]
                
                # Find all extrema points within this range
                search_extremas = [e for e in all_extremas if close_prices.index.get_loc(e) >= start_index and close_prices.index.get_loc(e) <= end_index]
                
                # Skip merged wave creation - just remove overlapping waves
                print(" -> 重叠波浪已移除（合并功能已禁用）。")

                # Remove both overlapping waves and skip next wave
                i += 2
                break
            else:
                j += 1
        
        if not overlap_found:
            final_waves.append(current_wave)
            i += 1
            
    return final_waves, merged_waves

# =========================================================
# Overall trend validation helper functions
# =========================================================

def is_overall_downtrend(close_prices, start_idx, end_idx, trend_threshold=0.95):
    """
    Check if overall trend from start to end is downward
    
    Parameters:
        close_prices (pd.Series): Price data
        start_idx (int): Starting index 
        end_idx (int): Ending index
        trend_threshold (float): Ratio threshold - if end_price/start_price < threshold, it's downtrend
    
    Returns:
        bool: True if overall trend is downward
    """
    if start_idx >= end_idx:
        return False
        
    start_price = close_prices.iloc[start_idx]
    end_price = close_prices.iloc[end_idx]
    price_ratio = end_price / start_price
    
    return price_ratio < trend_threshold

def validate_wave_overall_trend(close_prices, wave_indices, trend_threshold=0.95):
    """
    Validate that a wave occurs within an overall downtrend
    
    Parameters:
        close_prices (pd.Series): Price data
        wave_indices (list): Wave point indices
        trend_threshold (float): Trend validation threshold
    
    Returns:
        tuple: (is_valid, trend_ratio, trend_info)
    """
    start_idx = wave_indices[0]  # P1
    end_idx = wave_indices[-1]   # P6/P8
    
    start_price = close_prices.iloc[start_idx]
    end_price = close_prices.iloc[end_idx]
    trend_ratio = end_price / start_price
    
    is_downtrend = trend_ratio < trend_threshold
    
    trend_info = {
        'start_date': close_prices.index[start_idx].date(),
        'end_date': close_prices.index[end_idx].date(),
        'start_price': start_price,
        'end_price': end_price,
        'trend_ratio': trend_ratio,
        'is_downtrend': is_downtrend
    }
    
    return is_downtrend, trend_ratio, trend_info

# =========================================================
# Price refinement helper functions
# =========================================================

def refine_peak_with_price(close_prices, rsi_peak_date, window_days=5):
    """
    Refine RSI-confirmed peak using price data
    Look for higher price within window after RSI peak
    """
    try:
        peak_idx = close_prices.index.get_loc(rsi_peak_date)
        peak_price = close_prices.iloc[peak_idx]
        
        # Search window: look ahead from RSI peak
        end_idx = min(peak_idx + window_days + 1, len(close_prices))
        search_window = close_prices.iloc[peak_idx:end_idx]
        
        # Find highest price point in window
        max_price_idx = search_window.idxmax()
        max_price = search_window.max()
        
        # If higher price found, adjust peak point
        if max_price > peak_price:
            print(f"   -> 价格微调: 峰值从 {rsi_peak_date.date()} ({peak_price:.2f}) 调整到 {max_price_idx.date()} ({max_price:.2f})")
            return max_price_idx
        else:
            return rsi_peak_date
    except:
        return rsi_peak_date

def refine_valley_with_price(close_prices, rsi_valley_date, window_days=5):
    """
    Refine RSI-confirmed valley using price data
    Look for lower price within window after RSI valley
    """
    try:
        valley_idx = close_prices.index.get_loc(rsi_valley_date)
        valley_price = close_prices.iloc[valley_idx]
        
        # Search window: look ahead from RSI valley
        end_idx = min(valley_idx + window_days + 1, len(close_prices))
        search_window = close_prices.iloc[valley_idx:end_idx]
        
        # Find lowest price point in window
        min_price_idx = search_window.idxmin()
        min_price = search_window.min()
        
        # If lower price found, adjust valley point
        if min_price < valley_price:
            print(f"   -> 价格微调: 谷值从 {rsi_valley_date.date()} ({valley_price:.2f}) 调整到 {min_price_idx.date()} ({min_price:.2f})")
            return min_price_idx
        else:
            return rsi_valley_date
    except:
        return rsi_valley_date

# =========================================================
# RSI-driven extrema detection (Updated hybrid RSI-price method)
# =========================================================

def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3, rsi_rise_cap=None, price_refinement_window=5):
    """
    Find extrema points using RSI triggers with price refinement
    
    Parameters:
        close_prices (pd.Series): Price data with datetime index
        rsi_period (int): RSI calculation period
        rsi_drop_threshold (int): RSI drop threshold for peak detection
        rsi_rise_ratio (float): RSI rise ratio for valley detection (fraction of previous drop)
        rsi_rise_cap (float): Optional RSI rise cap - absolute RSI rise threshold regardless of drop size
        price_refinement_window (int): Window for price-based extrema refinement
    
    Returns:
        tuple: (extremas, trigger_points)
    """
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    extremas = []
    trigger_points = []
    
    print("\n--- Starting RSI-driven Extremas Identification ---")
    
    if len(rsi_series) < 2: 
        return extremas, trigger_points
    
    peak_rsi = rsi_series.iloc[0]
    valley_rsi = rsi_series.iloc[0]
    peak_date = rsi_series.index[0]
    valley_date = rsi_series.index[0]
    
    last_wave_peak_rsi = peak_rsi
    direction = 0
    
    for i in range(1, len(rsi_series)):
        current_rsi = rsi_series.iloc[i]
        current_date = rsi_series.index[i]
        
        if direction >= 0:
            if current_rsi > peak_rsi:
                peak_rsi = current_rsi
                peak_date = current_date
            
            rsi_drop_change = peak_rsi - current_rsi
            if rsi_drop_change >= rsi_drop_threshold:
                # Use price refinement for peak point
                refined_peak_date = refine_peak_with_price(close_prices, peak_date, price_refinement_window)
                
                if not extremas or extremas[-1] != refined_peak_date:
                    extremas.append(refined_peak_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'drop', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,
                    'change': rsi_drop_change,
                    'threshold': rsi_drop_threshold,
                    'refined_peak_date': refined_peak_date
                })
                
                print(f"[{current_date.date()}] Triggered RSI Drop Trigger!")
                print(f"   -> Confirmed Peak at: {peak_date.date()}")
                if refined_peak_date != peak_date:
                    print(f"   -> Price-adjusted peak: {refined_peak_date.date()}")
                print(f"   -> RSI dropped from high {peak_rsi:.2f} to {current_rsi:.2f}, total drop {rsi_drop_change:.2f} points, meeting threshold {rsi_drop_threshold}.")
                
                direction = -1
                last_wave_peak_rsi = peak_rsi
                valley_rsi = current_rsi
                valley_date = current_date
        
        if direction <= 0:
            if current_rsi < valley_rsi:
                valley_rsi = current_rsi
                valley_date = current_date
            
            rsi_rise_change = current_rsi - valley_rsi
            drop_amount_for_threshold = last_wave_peak_rsi - valley_rsi
            rise_threshold = drop_amount_for_threshold * rsi_rise_ratio
            
            # Check both rise ratio condition and optional rise cap condition
            ratio_condition = rsi_rise_change >= rise_threshold and drop_amount_for_threshold > 0
            cap_condition = rsi_rise_cap is not None and rsi_rise_change >= rsi_rise_cap
            
            if ratio_condition or cap_condition:
                # Use price refinement for valley point
                refined_valley_date = refine_valley_with_price(close_prices, valley_date, price_refinement_window)
                
                if not extremas or extremas[-1] != refined_valley_date:
                    extremas.append(refined_valley_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'rise', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,
                    'change': rsi_rise_change,
                    'threshold': rise_threshold,
                    'cap_threshold': rsi_rise_cap,
                    'triggered_by': 'cap' if cap_condition and not ratio_condition else 'ratio' if ratio_condition else 'both',
                    'refined_valley_date': refined_valley_date
                })
                
                print(f"[{current_date.date()}] Triggered RSI Rise Trigger!")
                print(f"   -> Confirmed Valley at: {valley_date.date()}")
                if refined_valley_date != valley_date:
                    print(f"   -> Price-adjusted valley: {refined_valley_date.date()}")
                
                # Show which condition triggered the rebound
                if cap_condition and ratio_condition:
                    print(f"   -> RSI rose from low {valley_rsi:.2f} to {current_rsi:.2f}, total rise {rsi_rise_change:.2f} points, meeting both ratio threshold {rise_threshold:.2f} and cap threshold {rsi_rise_cap}.")
                elif cap_condition:
                    print(f"   -> RSI rose from low {valley_rsi:.2f} to {current_rsi:.2f}, total rise {rsi_rise_change:.2f} points, meeting cap threshold {rsi_rise_cap} (ratio threshold was {rise_threshold:.2f}).")
                else:
                    print(f"   -> RSI rose from low {valley_rsi:.2f} to {current_rsi:.2f}, total rise {rsi_rise_change:.2f} points, meeting ratio threshold {rise_threshold:.2f}.")
                
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    print("\n--- RSI-driven Extremas Identification Finished ---")
    return extremas, trigger_points

# =========================================================
# Sequential RSI-based P0 Wave Detection (New Implementation)
# =========================================================

def find_continuous_waves_from_recent_highs(close_prices, recent_days=50, rsi_period=14, 
                                          rsi_drop_threshold=10, rsi_rise_ratio=1/3, 
                                          price_refinement_window=5, trend_threshold=0.95):
    """
    基于最近高RSI点的连续波浪检测新逻辑 - Updated Implementation
    
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
            rsi_rise_ratio=rsi_rise_ratio
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
    """
    Sequential wave detection using RSI-based P0 selection as specified in demand.md
    
    Parameters:
        close_prices (pd.Series): Price data with datetime index
        rsi_period (int): RSI calculation period
        lookback_days (int): Days to look back for highest RSI point (P0 selection)
        rsi_drop_threshold (int): RSI drop threshold for extrema detection
        rsi_rise_ratio (float): RSI rise ratio for extrema detection
        rsi_rise_cap (float): Optional RSI rise cap - absolute RSI rise threshold
        price_refinement_window (int): Window for price-based P0 refinement
        recent_days (int): Minimum gap between waves (when allow_overlaps=False)
        allow_overlaps (bool): If True, use smaller steps to allow overlapping waves
    
    Returns:
        tuple: (all_waves, all_trigger_points)
    """
    print(f"=== Sequential RSI-based P0 Wave Detection ===")
    print(f"Parameters: lookback_days={lookback_days}, recent_days={recent_days}, allow_overlaps={allow_overlaps}")
    print()
    
    # Calculate RSI for entire dataset
    rsi_series = calculate_rsi(close_prices, period=rsi_period)
    
    all_waves = []
    all_trigger_points = []
    current_position = 0  # Start from beginning of dataset
    
    wave_count = 0
    max_iterations = 50  # Prevent infinite loops
    
    # Determine advancement step based on overlap setting
    if allow_overlaps:
        advancement_step = lookback_days // 3  # Much smaller steps to allow overlaps
        print(f"Using overlap-friendly advancement step: {advancement_step} days")
    else:
        advancement_step = recent_days
        print(f"Using non-overlap advancement step: {advancement_step} days")
    
    while current_position < len(close_prices) - lookback_days and wave_count < max_iterations:
        wave_count += 1
        print(f"\n--- Wave Search Iteration {wave_count} ---")
        
        # Define search window: next lookback_days of unused data
        end_position = min(current_position + lookback_days, len(close_prices))
        search_data = close_prices.iloc[current_position:end_position]
        search_rsi = rsi_series.iloc[current_position:end_position]
        
        if len(search_data) < 10:  # Not enough data left
            print(f"Insufficient data remaining ({len(search_data)} points). Stopping search.")
            break
            
        print(f"Searching in range: {search_data.index[0].date()} to {search_data.index[-1].date()} ({len(search_data)} days)")
        
        # Find highest RSI point in this window as P0 candidate
        if search_rsi.isna().all():
            print("No valid RSI data in search window. Advancing position.")
            current_position += advancement_step
            continue
            
        # Find highest RSI point
        highest_rsi_idx_local = search_rsi.idxmax()
        highest_rsi_value = search_rsi.max()
        
        if pd.isna(highest_rsi_value):
            print("No valid RSI peak found. Advancing position.")
            current_position += advancement_step
            continue
            
        print(f"Highest RSI in window: {highest_rsi_value:.2f} at {highest_rsi_idx_local.date()}")
        
        # Apply price refinement to P0
        p0_date = refine_peak_with_price(close_prices, highest_rsi_idx_local, price_refinement_window)
        p0_price = close_prices.loc[p0_date]
        
        if p0_date != highest_rsi_idx_local:
            print(f"P0 refined to: {p0_date.date()} (${p0_price:.2f})")
        
        # Now find a complete wave starting from this P0
        # Create a subset of data starting from P0 for wave detection
        p0_position = close_prices.index.get_loc(p0_date)
        wave_search_end = min(p0_position + 200, len(close_prices))  # Look ahead 200 days max for wave completion
        wave_search_data = close_prices.iloc[p0_position:wave_search_end]
        
        if len(wave_search_data) < 30:  # Need sufficient data for wave detection
            print("Insufficient data for wave detection from P0. Advancing position.")
            current_position += advancement_step
            continue
            
        # Use existing RSI-driven extrema detection starting from P0
        wave_extremas, wave_triggers = find_extremas_with_rsi(
            wave_search_data, 
            rsi_period=rsi_period,
            rsi_drop_threshold=rsi_drop_threshold, 
            rsi_rise_ratio=rsi_rise_ratio,
            rsi_rise_cap=rsi_rise_cap,
            price_refinement_window=price_refinement_window
        )
        
        # Ensure P0 is included as first extrema
        if not wave_extremas or wave_extremas[0] != p0_date:
            wave_extremas.insert(0, p0_date)
            
        if len(wave_extremas) < 6:  # Need at least 6 points for a 5-wave pattern
            print(f"Only found {len(wave_extremas)} extrema points. Need at least 6 for wave pattern.")
            current_position += advancement_step
            continue
            
        # Try to find wave patterns from these extremas with trend validation
        wave_patterns = find_downtrend_wave_patterns(close_prices, wave_extremas, length=6, trend_threshold=0.95)
        
        if wave_patterns:
            # Found at least one wave pattern
            best_wave = wave_patterns[0]  # Take the first valid wave
            wave_start_idx = best_wave['indices'][0]
            wave_end_idx = best_wave['indices'][-1]
            
            wave_start_date = close_prices.index[wave_start_idx]
            wave_end_date = close_prices.index[wave_end_idx]
            
            print(f"✓ Found {best_wave['type']} wave: {wave_start_date.date()} to {wave_end_date.date()}")
            
            all_waves.append(best_wave)
            all_trigger_points.extend(wave_triggers)
            
            # Choose advancement strategy
            if allow_overlaps:
                # Small step to potentially find overlapping waves
                current_position += advancement_step
            else:
                # Advance to after this wave + minimum gap (original behavior)
                current_position = wave_end_idx + recent_days
        else:
            print("No valid wave pattern found from P0. Advancing position.")
            current_position += advancement_step
    
    print(f"\n=== Sequential Wave Search Complete ===")
    print(f"Total waves found: {len(all_waves)}")
    
    return all_waves, all_trigger_points

def get_benchmark_wave_count():
    """
    从benchmark.txt文件中解析基准波浪数量
    
    返回:
        int: 基准波浪总数
    """
    try:
        with open('benchmark.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 查找总波浪数量的行
        import re
        lines = content.split('\n')
        for line in lines:
            if '原始识别:' in line and '个波浪结构' in line:
                # 提取数字
                match = re.search(r'(\d+)\s*个波浪结构', line)
                if match:
                    original_count = int(match.group(1))
                    
            if '处理后:' in line and '个独立波浪' in line and '个合并波浪' in line:
                # 提取独立和合并波浪数量
                matches = re.findall(r'(\d+)\s*个', line)
                if len(matches) >= 2:
                    independent_count = int(matches[0])
                    merged_count = int(matches[1])
                    return independent_count + merged_count
                    
        return 0
        
    except FileNotFoundError:
        print("基准文件 benchmark.txt 未找到")
        return 0
    except Exception as e:
        print(f"解析基准文件时出错: {e}")
        return 0

# =========================================================
# Main analysis function - Enhanced RSI P0 Selection Version
# =========================================================

def main(file_path='BTC.csv', recent_days=50, rsi_period=14, rsi_drop_threshold=10, 
         rsi_rise_ratio=1/3, price_refinement_window=5, trend_threshold=0.95):
    """
    Enhanced wave analysis using RSI-based P0 selection
    
    参数:
        file_path (str): 数据文件路径
        recent_days (int): P0搜索回望天数和最小间隔
        rsi_period (int): RSI计算周期
        rsi_drop_threshold (int): RSI下跌阈值  
        rsi_rise_ratio (float): RSI上涨比例
        price_refinement_window (int): 价格微调窗口
    
    返回:
        dict: 分析结果字典
    """
    print("=== BTC 波浪检测系统 (Enhanced RSI P0 Selection) ===")
    print("基于最高RSI点起始的连续波浪检测系统")
    print("支持智能P0选择、重叠波浪合并和基准质量验证")
    print()
    
    # 加载数据
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None
        
    if 'datetime' not in data.columns or 'close' not in data.columns:
        print("错误: CSV文件必须包含 'datetime' 和 'close' 两列。")
        return None
        
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    close_prices = data['close']
    
    print(f"成功加载 {len(close_prices)} 条数据记录")
    print(f"数据时间范围: {close_prices.index.min().date()} 到 {close_prices.index.max().date()}")
    print()
    
    # 计算RSI
    print("正在计算 RSI 指标...")
    rsi_series = calculate_rsi(close_prices, period=rsi_period)
    print("RSI 计算完成")
    print()
    
    # 使用新的基于高RSI点的连续波浪检测逻辑
    print("正在执行基于最高RSI点的连续波浪检测...")
    all_waves, all_trigger_points = find_continuous_waves_from_recent_highs(
        close_prices, 
        recent_days=recent_days,
        rsi_period=rsi_period, 
        rsi_drop_threshold=rsi_drop_threshold, 
        rsi_rise_ratio=rsi_rise_ratio,
        price_refinement_window=price_refinement_window,
        trend_threshold=trend_threshold
    )
    
    # 处理重叠波浪
    if all_waves:
        print("\n正在处理重叠波浪...")
        # 为了兼容现有的overlap处理函数，我们需要提取所有极值点
        all_extremas_set = set()
        for wave in all_waves:
            for idx in wave['indices']:
                all_extremas_set.add(close_prices.index[idx])
        all_extremas = sorted(list(all_extremas_set))
        
        final_waves, merged_waves = handle_overlapping_waves(all_waves, close_prices, all_extremas)
        
        print(f"原始识别: {len(all_waves)} 个波浪结构")
        print(f"处理后: {len(final_waves)} 个独立波浪，{len(merged_waves)} 个合并波浪")
    else:
        final_waves, merged_waves = [], []
        print("未检测到有效波浪结构")
    
    # 波浪检测完成
    total_waves = len(final_waves) + len(merged_waves)
    print(f"\n=== 波浪检测完成 ===")
    print(f"总共识别出 {total_waves} 个有效波浪结构")
    
    # 基准验证
    benchmark_count = get_benchmark_wave_count()
    if benchmark_count > 0:
        detection_rate = total_waves / benchmark_count
        print(f"\n=== 基准验证 ===")
        print(f"检测到波浪数: {total_waves}")
        print(f"基准波浪数: {benchmark_count}")
        print(f"检测率: {detection_rate:.2%}")
        
        if detection_rate < 0.8:
            print("⚠️  警告: 检测率低于80%，建议调整算法参数或逻辑")
        else:
            print("✅ 检测率满足质量要求 (≥80%)")
    
    # 返回分析结果
    return {
        'close_prices': close_prices,
        'rsi_series': rsi_series,
        'final_waves': final_waves,
        'merged_waves': merged_waves,
        'total_waves': total_waves,
        'benchmark_count': benchmark_count,
        'detection_rate': detection_rate if benchmark_count > 0 else None
    }

# =========================================================
# Enhanced plotting function (pandas compatibility fixes)
# =========================================================

def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type=""):
    """
    Plot individual wave with RSI subplot (pandas compatible)
    """
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    plot_rsi = rsi_series.loc[start_date:end_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    ax1.plot(plot_prices.index, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'magenta'
    ax1.plot(wave_points_dates, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    # Dynamic point annotations
    for j in range(len(wave_indices)):
        ax1.annotate(f'P{j+1}', (wave_points_dates[j], wave_points_prices[j]), xytext=(5, 5), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
        
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            trigger_price = close_prices.loc[point['date']]
            ax1.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            marker = 'v' if point['type'] == 'drop' else '^'
            label = 'RSI Drop Trigger' if point['type'] == 'drop' else 'RSI Rise Trigger'
            ax1.plot(point['date'], trigger_price, marker, color='orange', markersize=8, label=label)
            
    ax1.set_title(f'BTC {wave_type.capitalize()} Wave {wave_number}: {wave_start_date.date()} to {wave_end_date.date()}', fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys())
    ax1.grid(True)
    
    ax2.plot(plot_rsi.index, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
    wave_points_rsi = rsi_series.loc[wave_points_dates].values
    ax2.plot(wave_points_dates, wave_points_rsi, 'o', color=color, markersize=6)
    for j in range(len(wave_indices)):
        ax2.annotate(f'P{j+1} ({wave_points_rsi[j]:.2f})', (wave_points_dates[j], wave_points_rsi[j]), xytext=(5, 5), textcoords='offset points', fontsize=8, color=color, fontweight='bold')
    
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            marker = 'v' if point['type'] == 'drop' else '^'
            ax2.plot(point['date'], point['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            ax2.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            
            rsi_value_text = f"{point['rsi_value']:.2f}"
            y_offset = -10 if point['type'] == 'drop' else 10
            ax2.annotate(rsi_value_text, (point['date'], point['rsi_value']),
                         xytext=(0, y_offset), textcoords='offset points',
                         ha='center', va='center', fontsize=10, color='orange',
                         bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
            
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    
    plt.tight_layout()
    plt.show()

def plot_overview_chart(close_prices, all_waves, start_date_all, end_date_all, rsi_drop_threshold, rsi_rise_ratio):
    """
    Plot overview chart showing all detected waves (pandas compatible)
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot Close Price first
    ax.plot(close_prices.index.values, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)

    # Flags to plot a single label for each wave type
    plotted_merged_label = False
    plotted_strict_label = False
    plotted_relaxed_label = False

    if 'merged_waves' in all_waves and all_waves['merged_waves']:
        print(f"\n成功识别出 {len(all_waves['merged_waves'])} 个合并波浪结构。")
        for i, wave in enumerate(all_waves['merged_waves']):
            wave_indices = wave['indices']
            wave_points_dates = close_prices.index[wave_indices]
            label = 'Merged Wave' if not plotted_merged_label else None
            ax.plot(wave_points_dates, close_prices.iloc[wave_indices].values, 'ms-', markersize=6, label=label)
            plotted_merged_label = True
            print(f"   - 合并波浪 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax.annotate(f'Merged Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color='magenta', fontweight='bold')

    if 'final_waves' in all_waves and all_waves['final_waves']:
        print(f"\n成功识别出 {len(all_waves['final_waves'])} 个不重叠的波浪结构。")
        for i, wave in enumerate(all_waves['final_waves']):
            wave_indices = wave['indices']
            wave_type = wave['type']
            wave_points_dates = close_prices.index[wave_indices]
            color = 'red' if wave_type == 'strict' else 'green'
            
            if wave_type == 'strict' and not plotted_strict_label:
                label = 'Strict Wave'
                plotted_strict_label = True
            elif wave_type == 'relaxed' and not plotted_relaxed_label:
                label = 'Relaxed Wave'
                plotted_relaxed_label = True
            else:
                label = None

            ax.plot(wave_points_dates, close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
            
            print(f"   - {wave_type.capitalize()} 波浪 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax.annotate(f'{wave_type.capitalize()} Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
            
    ax.legend(loc='upper right')

    title_str = f'BTC Price Chart: {start_date_all} to {end_date_all} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}'
    ax.set_title(title_str, fontsize=16)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    plt.tight_layout()
    plt.show()

def run_wave_analysis(file_path='BTC.csv', recent_days=50, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3, price_refinement_window=5, trend_threshold=0.95):
    """
    Execute complete wave analysis workflow - Enhanced RSI P0 Selection Version
    """
    print("=== BTC 波浪检测系统 (Enhanced RSI P0 Selection) ===")
    print("基于最高RSI点起始的连续波浪检测系统")
    print("支持智能P0选择、重叠波浪合并和基准质量验证")
    print()
    
    # Load data
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return None
        
    if 'datetime' not in data.columns or 'close' not in data.columns:
        print("错误: CSV文件必须包含 'datetime' 和 'close' 两列。")
        return None
        
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    close_prices = data['close']
    
    print(f"成功加载 {len(close_prices)} 条数据记录")
    print(f"数据时间范围: {close_prices.index.min().date()} 到 {close_prices.index.max().date()}")
    print()
    
    # Calculate RSI
    print("正在计算 RSI 指标...")
    rsi_series = calculate_rsi(close_prices, period=rsi_period)
    print("RSI 计算完成")
    print()
    
    # Use enhanced RSI P0 selection logic
    print("正在执行基于最高RSI点的连续波浪检测...")
    all_waves, all_trigger_points = find_continuous_waves_from_recent_highs(
        close_prices, 
        recent_days=recent_days,
        rsi_period=rsi_period, 
        rsi_drop_threshold=rsi_drop_threshold, 
        rsi_rise_ratio=rsi_rise_ratio,
        price_refinement_window=price_refinement_window,
        trend_threshold=trend_threshold
    )
    
    # Handle overlapping waves
    if all_waves:
        print("\n正在处理重叠波浪...")
        # Extract all extremas for compatibility
        all_extremas_set = set()
        for wave in all_waves:
            for idx in wave['indices']:
                all_extremas_set.add(close_prices.index[idx])
        all_extremas = sorted(list(all_extremas_set))
        
        final_waves, merged_waves = handle_overlapping_waves(all_waves, close_prices, all_extremas)
        
        print(f"原始识别: {len(all_waves)} 个波浪结构")
        print(f"处理后: {len(final_waves)} 个独立波浪，{len(merged_waves)} 个合并波浪")
    else:
        final_waves, merged_waves = [], []
        print("未检测到有效波浪结构")
    
    total_waves = len(final_waves) + len(merged_waves)
    print(f"\n=== 波浪检测完成 ===")
    print(f"总共识别出 {total_waves} 个有效波浪结构")
    
    # Benchmark validation
    benchmark_count = get_benchmark_wave_count()
    if benchmark_count > 0:
        detection_rate = total_waves / benchmark_count
        print(f"\n=== 基准验证 ===")
        print(f"检测到波浪数: {total_waves}")
        print(f"基准波浪数: {benchmark_count}")
        print(f"检测率: {detection_rate:.2%}")
        
        if detection_rate < 0.8:
            print("⚠️  警告: 检测率低于80%，建议调整算法参数或逻辑")
        else:
            print("✅ 检测率满足质量要求 (≥80%)")
    
    # Plot overview
    wave_data = {
        'final_waves': final_waves,
        'merged_waves': merged_waves
    }
    
    try:
        plot_overview_chart(
            close_prices, 
            wave_data,
            close_prices.index.min().date(), 
            close_prices.index.max().date(), 
            rsi_drop_threshold, 
            rsi_rise_ratio
        )
    except Exception as e:
        print(f"绘图出错: {e}")
        print("跳过绘图，继续分析...")
    
    # Plot individual waves
    all_processed_waves = final_waves + merged_waves
    if all_processed_waves:
        print(f"\n正在为每个识别出的波浪生成独立的放大图...")
        for i, wave in enumerate(all_processed_waves):
            wave_indices = wave['indices']
            wave_type = wave['type']
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            try:
                plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=15, wave_number=i+1, wave_type=wave_type)
            except Exception as e:
                print(f"绘制波浪 {i+1} 时出错: {e}")
                continue
    
    return {
        'close_prices': close_prices,
        'rsi_series': rsi_series,
        'final_waves': final_waves,
        'merged_waves': merged_waves,
        'total_waves': total_waves,
        'benchmark_count': benchmark_count,
        'detection_rate': detection_rate if benchmark_count > 0 else None
    }

if __name__ == "__main__":
    # Enhanced RSI P0 Selection Analysis
    results = run_wave_analysis(
        file_path='BTC.csv',
        recent_days=50,
        rsi_period=14,
        rsi_drop_threshold=10,
        rsi_rise_ratio=1/3,
        price_refinement_window=5,
        trend_threshold=0.95
    )