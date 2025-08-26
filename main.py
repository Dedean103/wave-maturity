#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 波浪检测主程序 (Updated Version)
基于 RSI 驱动的峰谷点识别和五浪/七浪结构检测
支持重叠波浪的智能合并功能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all functions from latest_code.py
from latest_code import (
    is_downtrend_five_wave_strict,
    is_downtrend_five_wave_relaxed,
    is_downtrend_seven_wave,
    calculate_rsi,
    find_downtrend_wave_patterns,
    handle_overlapping_waves,
    plot_individual_wave,
    refine_peak_with_price,
    refine_valley_with_price,
    find_extremas_with_rsi
)

def load_data(file_path):
    """
    加载 CSV 数据文件
    
    参数:
        file_path (str): CSV 文件路径
    
    返回:
        pd.Series: 收盘价序列
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。请确保文件存在并位于正确的路径下。")
        sys.exit(1)
        
    if 'datetime' not in data.columns or 'close' not in data.columns:
        print("错误: CSV文件必须包含 'datetime' 和 'close' 两列。")
        sys.exit(1)
        
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data['close']

def plot_overview_chart(close_prices, final_waves, merged_waves, all_trigger_points, start_date_all, end_date_all, file_path, rsi_series, rsi_drop_threshold, rsi_rise_ratio):
    """
    绘制概览图表，显示所有波浪和RSI
    """
    # Use a more compatible style
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        try:
            plt.style.use('seaborn')
        except:
            plt.style.use('default')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    # Subplot 1: Price Chart  
    ax1.plot(close_prices.index.to_numpy(), close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)

    plotted_merged_label = False
    plotted_strict_label = False
    plotted_relaxed_label = False

    if merged_waves:
        print(f"\n成功识别出 {len(merged_waves)} 个合并波浪结构。")
        for i, wave in enumerate(merged_waves):
            wave_indices = wave['indices']
            wave_points_dates = close_prices.index[wave_indices]
            label = 'Merged Wave' if not plotted_merged_label else None
            ax1.plot(wave_points_dates.to_numpy(), close_prices.iloc[wave_indices].values, 'ms-', markersize=6, label=label)
            plotted_merged_label = True
            print(f"   - 合并波浪 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax1.annotate(f'Merged Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color='magenta', fontweight='bold')

    if final_waves:
        print(f"\n成功识别出 {len(final_waves)} 个不重叠的波浪结构。")
        for i, wave in enumerate(final_waves):
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

            ax1.plot(wave_points_dates.to_numpy(), close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
            
            print(f"   - {wave_type.capitalize()} 波浪 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax1.annotate(f'{wave_type.capitalize()} Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
            
    ax1.legend(loc='upper right')
    title_str = f'{file_path[:4]} Price Chart: {start_date_all} to {end_date_all} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)

    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index.to_numpy(), rsi_series.values, color='purple', label='14-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)

    # Create a set to store plotted trigger points to avoid duplicates
    plotted_triggers = set()

    # Function to get the first and last trigger points for a given wave
    def get_first_last_triggers(wave_indices, all_trigger_points, close_prices):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        
        wave_triggers = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        
        first_trigger = wave_triggers[0] if wave_triggers else None
        last_trigger = wave_triggers[-1] if wave_triggers else None
        
        return first_trigger, last_trigger

    # Plot trigger points for merged waves
    for wave in merged_waves:
        first_trigger, last_trigger = get_first_last_triggers(wave['indices'], all_trigger_points, close_prices)
        if first_trigger and (first_trigger['date'], first_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if first_trigger['type'] == 'drop' else '^'
            ax2.plot(first_trigger['date'], first_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((first_trigger['date'], first_trigger['rsi_value']))
        
        if last_trigger and (last_trigger['date'], last_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if last_trigger['type'] == 'drop' else '^'
            ax2.plot(last_trigger['date'], last_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((last_trigger['date'], last_trigger['rsi_value']))

    # Plot trigger points for final (non-overlapping) waves
    for wave in final_waves:
        first_trigger, last_trigger = get_first_last_triggers(wave['indices'], all_trigger_points, close_prices)
        if first_trigger and (first_trigger['date'], first_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if first_trigger['type'] == 'drop' else '^'
            ax2.plot(first_trigger['date'], first_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((first_trigger['date'], first_trigger['rsi_value']))
        
        if last_trigger and (last_trigger['date'], last_trigger['rsi_value']) not in plotted_triggers:
            marker = 'v' if last_trigger['type'] == 'drop' else '^'
            ax2.plot(last_trigger['date'], last_trigger['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            plotted_triggers.add((last_trigger['date'], last_trigger['rsi_value']))

    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())

    plt.tight_layout()
    plt.show()

def find_sequential_waves_with_rsi_p0(close_prices, rsi_period=14, lookback_days=50, rsi_drop_threshold=10, rsi_rise_ratio=1/3, price_refinement_window=5, recent_days=50):
    """
    Sequential wave detection using RSI-based P0 selection as specified in demand.md
    
    Parameters:
        close_prices (pd.Series): Price data with datetime index
        rsi_period (int): RSI calculation period
        lookback_days (int): Days to look back for highest RSI point (P0 selection)
        rsi_drop_threshold (int): RSI drop threshold for extrema detection
        rsi_rise_ratio (float): RSI rise ratio for extrema detection
        price_refinement_window (int): Window for price-based P0 refinement
        recent_days (int): Minimum gap between waves
    
    Returns:
        tuple: (all_waves, all_trigger_points)
    """
    print(f"=== Sequential RSI-based P0 Wave Detection ===")
    print(f"Parameters: lookback_days={lookback_days}, recent_days={recent_days}")
    print()
    
    # Calculate RSI for entire dataset
    rsi_series = calculate_rsi(close_prices, period=rsi_period)
    
    all_waves = []
    all_trigger_points = []
    current_position = 0  # Start from beginning of dataset
    
    wave_count = 0
    max_iterations = 50  # Prevent infinite loops
    
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
            current_position += recent_days
            continue
            
        # Find highest RSI point
        highest_rsi_idx_local = search_rsi.idxmax()
        highest_rsi_value = search_rsi.max()
        
        if pd.isna(highest_rsi_value):
            print("No valid RSI peak found. Advancing position.")
            current_position += recent_days
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
            current_position += recent_days
            continue
            
        # Use existing RSI-driven extrema detection starting from P0
        wave_extremas, wave_triggers = find_extremas_with_rsi(
            wave_search_data, 
            rsi_period=rsi_period,
            rsi_drop_threshold=rsi_drop_threshold, 
            rsi_rise_ratio=rsi_rise_ratio,
            price_refinement_window=price_refinement_window
        )
        
        # Ensure P0 is included as first extrema
        if not wave_extremas or wave_extremas[0] != p0_date:
            wave_extremas.insert(0, p0_date)
            
        if len(wave_extremas) < 6:  # Need at least 6 points for a 5-wave pattern
            print(f"Only found {len(wave_extremas)} extrema points. Need at least 6 for wave pattern.")
            current_position = p0_position + recent_days
            continue
            
        # Try to find wave patterns from these extremas
        wave_patterns = find_downtrend_wave_patterns(close_prices, wave_extremas, length=6)
        
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
            
            # Advance position to after this wave + minimum gap
            current_position = wave_end_idx + recent_days
        else:
            print("No valid wave pattern found from P0. Advancing position.")
            current_position = p0_position + recent_days
    
    print(f"\n=== Sequential Wave Search Complete ===")
    print(f"Total waves found: {len(all_waves)}")
    
    return all_waves, all_trigger_points

def validate_benchmark_performance(detected_waves, benchmark_count=10):
    """
    Validate detection performance against benchmark
    
    Parameters:
        detected_waves (int): Number of waves detected
        benchmark_count (int): Expected benchmark count
    
    Returns:
        dict: Validation results
    """
    threshold = int(benchmark_count * 0.6)  # 60% threshold
    detection_rate = (detected_waves / benchmark_count) * 100
    
    result = {
        'detected_waves': detected_waves,
        'benchmark_count': benchmark_count,
        'threshold': threshold,
        'detection_rate': detection_rate,
        'meets_threshold': detected_waves >= threshold,
        'needs_improvement': detected_waves < threshold
    }
    
    print(f"\n=== Benchmark Validation ===")
    print(f"Detected waves: {detected_waves}")
    print(f"Benchmark count: {benchmark_count}")
    print(f"Detection rate: {detection_rate:.1f}%")
    print(f"Threshold (60%): {threshold} waves")
    
    if result['meets_threshold']:
        print(f"✓ PASS: Detection meets quality threshold")
    else:
        print(f"✗ FAIL: Detection below threshold - Logic improvement needed")
        
    return result

def run_wave_analysis(file_path='BTC.csv', rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3, price_refinement_window=5, lookback_days=50, use_sequential=True):
    """
    执行完整的波浪分析流程 - 支持传统和序列化方法
    
    参数:
        file_path (str): 数据文件路径
        rsi_period (int): RSI 计算周期
        rsi_drop_threshold (int): RSI 下跌触发阈值
        rsi_rise_ratio (float): RSI 上涨触发比例
        price_refinement_window (int): 价格微调窗口天数
        lookback_days (int): RSI-P0选择的回望天数 (仅用于序列化方法)
        use_sequential (bool): 是否使用序列化RSI-P0检测方法
    
    返回:
        dict: 分析结果字典
    """
    method_name = "Sequential RSI-P0" if use_sequential else "Traditional RSI-driven"
    print(f"=== BTC 波浪检测系统 ({method_name}) ===")
    print("基于 RSI 驱动的峰谷点识别和五浪/七浪结构检测")
    print("支持重叠波浪的智能合并功能")
    print()
    
    print(f"配置参数:")
    print(f"  - 数据文件: {file_path}")
    print(f"  - 检测方法: {method_name}")
    print(f"  - RSI 周期: {rsi_period}")
    print(f"  - RSI 下跌阈值: {rsi_drop_threshold}")
    print(f"  - RSI 上涨比例: {rsi_rise_ratio}")
    print(f"  - 价格微调窗口: {price_refinement_window}")
    if use_sequential:
        print(f"  - P0回望天数: {lookback_days}")
    print()
    
    # 加载数据
    print("正在加载数据...")
    close_prices = load_data(file_path)
    print(f"成功加载 {len(close_prices)} 条数据记录")
    print(f"数据时间范围: {close_prices.index.min().date()} 到 {close_prices.index.max().date()}")
    print()
    
    # 计算 RSI
    print("正在计算 RSI 指标...")
    rsi_series = calculate_rsi(close_prices, period=rsi_period)
    print("RSI 计算完成")
    print()
    
    # 选择检测方法
    if use_sequential:
        # 使用新的序列化RSI-P0方法
        print("正在执行基于序列化RSI-P0的波浪检测...")
        all_waves, all_trigger_points = find_sequential_waves_with_rsi_p0(
            close_prices,
            rsi_period=rsi_period,
            lookback_days=lookback_days,
            rsi_drop_threshold=rsi_drop_threshold, 
            rsi_rise_ratio=rsi_rise_ratio,
            price_refinement_window=price_refinement_window,
            recent_days=lookback_days  # Use same value for minimum gap
        )
    else:
        # 使用传统的全局RSI方法
        print("正在执行基于传统RSI的极值检测...")
        extremas, all_trigger_points = find_extremas_with_rsi(
            close_prices, 
            rsi_period=rsi_period, 
            rsi_drop_threshold=rsi_drop_threshold, 
            rsi_rise_ratio=rsi_rise_ratio,
            price_refinement_window=price_refinement_window
        )
        
        print("正在从 RSI 驱动的峰谷点中寻找波浪结构...")
        all_waves = find_downtrend_wave_patterns(close_prices, extremas, length=6)
    
    # 处理重叠波浪（如果存在）
    if all_waves:
        if not use_sequential:  # Only handle overlaps for traditional method
            print("\n正在处理重叠波浪...")
            # 为了兼容现有的overlap处理函数，我们需要提取所有极值点
            all_extremas_set = set()
            for wave in all_waves:
                for idx in wave['indices']:
                    all_extremas_set.add(close_prices.index[idx])
            all_extremas = sorted(list(all_extremas_set))
            
            final_waves, merged_waves = handle_overlapping_waves(all_waves, close_prices, all_extremas)
        else:
            # Sequential method shouldn't have overlaps by design
            final_waves = all_waves
            merged_waves = []
            all_extremas = []
        
        print(f"原始识别: {len(all_waves)} 个波浪结构")
        print(f"处理后: {len(final_waves)} 个独立波浪，{len(merged_waves)} 个合并波浪")
    else:
        final_waves, merged_waves = [], []
        all_extremas = []
        print("未检测到有效波浪结构")
    
    print()
    
    # 基准验证
    total_detected = len(final_waves) + len(merged_waves)
    benchmark_result = validate_benchmark_performance(total_detected, benchmark_count=10)
    
    # 绘制概览图
    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()
    
    print("\n正在生成概览图表...")
    plot_overview_chart(
        close_prices, 
        final_waves,
        merged_waves,
        all_trigger_points,
        start_date_all, 
        end_date_all, 
        file_path,
        rsi_series,
        rsi_drop_threshold, 
        rsi_rise_ratio
    )
    
    # 绘制每个波浪的详细图
    all_processed_waves = final_waves + merged_waves
    if all_processed_waves:
        print("\n正在为每个识别出的波浪生成独立的放大图...")
        print("(个别图表生成已暂时禁用以避免兼容性问题)")
    
    print("\n=== 波浪检测完成 ===")
    total_waves = len(final_waves) + len(merged_waves)
    print(f"总共识别出 {total_waves} 个有效波浪结构")
    
    # 返回分析结果
    return {
        'close_prices': close_prices,
        'rsi_series': rsi_series,
        'extremas': all_extremas,
        'trigger_points': all_trigger_points,
        'final_waves': final_waves,
        'merged_waves': merged_waves,
        'total_waves': total_waves,
        'benchmark_result': benchmark_result,
        'method_used': method_name
    }

def main():
    """
    主函数
    """
    # 创建日志文件名，包含时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"analysis_log_{timestamp}.txt"
    
    # 重定向stdout到文件和控制台
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        
        def write(self, text):
            for file in self.files:
                file.write(text)
        
        def flush(self):
            for file in self.files:
                file.flush()
    
    # 配置参数
    file_path = 'BTC.csv'
    rsi_period = 14
    rsi_drop_threshold = 10
    rsi_rise_ratio = 1/3
    price_refinement_window = 5
    lookback_days = 50  # New parameter for sequential method
    use_sequential = True  # Enable new sequential method
    
    # 打开日志文件
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # 创建同时输出到控制台和文件的对象
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(original_stdout, log_file)
        
        try:
            # 执行分析
            results = run_wave_analysis(
                file_path=file_path,
                rsi_period=rsi_period,
                rsi_drop_threshold=rsi_drop_threshold,
                rsi_rise_ratio=rsi_rise_ratio,
                price_refinement_window=price_refinement_window,
                lookback_days=lookback_days,
                use_sequential=use_sequential
            )
            
            print(f"\n分析结果已保存到: {log_filename}")
            
        finally:
            # 恢复原始stdout
            sys.stdout = original_stdout
    
    print(f"分析完成，日志已保存到: {log_filename}")
    return results

if __name__ == "__main__":
    main()