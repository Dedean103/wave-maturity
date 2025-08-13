#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 波浪检测主程序
基于 RSI 驱动的峰谷点识别和五浪结构检测
"""

import pandas as pd
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.wave_detection import find_all_wave_patterns, remove_overlapping_waves
from core.rsi_analysis import find_extremas_with_rsi
from utils.technical_indicators import calculate_rsi
from visualization.plotting import plot_individual_wave, plot_overview_chart

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

def main():
    """
    主函数
    """
    print("=== BTC 波浪检测系统 ===")
    print("基于 RSI 驱动的峰谷点识别和五浪结构检测")
    print()
    
    # 配置参数
    file_path = 'BTC.csv'  # 可以修改为其他文件路径
    rsi_period = 14
    rsi_drop_threshold = 10
    rsi_rise_ratio = 1/4
    
    print(f"配置参数:")
    print(f"  - 数据文件: {file_path}")
    print(f"  - RSI 周期: {rsi_period}")
    print(f"  - RSI 下跌阈值: {rsi_drop_threshold}")
    print(f"  - RSI 上涨比例: {rsi_rise_ratio}")
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
    
    # 识别极值点
    print("正在识别 RSI 驱动的峰谷点...")
    extremas, all_trigger_points = find_extremas_with_rsi(
        close_prices, 
        rsi_period=rsi_period, 
        rsi_drop_threshold=rsi_drop_threshold, 
        rsi_rise_ratio=rsi_rise_ratio
    )
    print(f"识别出 {len(extremas)} 个极值点")
    print(f"识别出 {len(all_trigger_points)} 个触发点")
    print()
    
    # 寻找波浪结构
    print("正在从 RSI 驱动的峰谷点中寻找波浪结构...")
    strict_waves_all, relaxed_waves_all = find_all_wave_patterns(close_prices, extremas)
    
    # 移除重叠波浪
    strict_waves = remove_overlapping_waves(strict_waves_all, close_prices)
    relaxed_waves = remove_overlapping_waves(relaxed_waves_all, close_prices)
    
    print(f"识别出 {len(strict_waves_all)} 个严格波浪结构，去重后 {len(strict_waves)} 个")
    print(f"识别出 {len(relaxed_waves_all)} 个宽松波浪结构，去重后 {len(relaxed_waves)} 个")
    print()
    
    # 绘制概览图
    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()
    
    print("正在生成概览图表...")
    plot_overview_chart(
        close_prices, 
        strict_waves, 
        relaxed_waves, 
        start_date_all, 
        end_date_all, 
        rsi_drop_threshold, 
        rsi_rise_ratio
    )
    
    # 绘制严格波浪详细图
    if strict_waves:
        print("\n正在为每个识别出的严格波浪生成独立的放大图...")
        for i, wave_indices in enumerate(strict_waves):
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            
            plot_individual_wave(
                close_prices, 
                rsi_series, 
                wave_indices, 
                wave_trigger_points, 
                plot_range_days=15, 
                wave_number=i+1, 
                wave_type='strict'
            )
    
    # 绘制宽松波浪详细图
    if relaxed_waves:
        print("\n正在为每个识别出的宽松波浪生成独立的放大图...")
        for i, wave_indices in enumerate(relaxed_waves):
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            
            plot_individual_wave(
                close_prices, 
                rsi_series, 
                wave_indices, 
                wave_trigger_points, 
                plot_range_days=15, 
                wave_number=i+1, 
                wave_type='relaxed'
            )
    
    print("\n=== 波浪检测完成 ===")
    print(f"总共识别出 {len(strict_waves) + len(relaxed_waves)} 个有效波浪结构")

if __name__ == "__main__":
    main()
