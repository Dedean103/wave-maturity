#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTC 波浪检测主程序 (Updated Version)
基于 RSI 驱动的峰谷点识别和五浪/七浪结构检测
支持重叠波浪的智能合并功能
"""

import pandas as pd
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.wave_detection import find_downtrend_wave_patterns, handle_overlapping_waves
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

def run_wave_analysis(file_path='BTC.csv', rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3):
    """
    执行完整的波浪分析流程
    
    参数:
        file_path (str): 数据文件路径
        rsi_period (int): RSI 计算周期
        rsi_drop_threshold (int): RSI 下跌触发阈值
        rsi_rise_ratio (float): RSI 上涨触发比例
    
    返回:
        dict: 分析结果字典
    """
    print("=== BTC 波浪检测系统 (Enhanced Version) ===")
    print("基于 RSI 驱动的峰谷点识别和五浪/七浪结构检测")
    print("支持重叠波浪的智能合并功能")
    print()
    
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
    all_waves = find_downtrend_wave_patterns(close_prices, extremas, length=6)
    
    # 处理重叠波浪
    print("正在处理重叠波浪...")
    final_waves, merged_waves = handle_overlapping_waves(all_waves, close_prices, extremas)
    
    print(f"原始识别: {len(all_waves)} 个波浪结构")
    print(f"处理后: {len(final_waves)} 个独立波浪，{len(merged_waves)} 个合并波浪")
    print()
    
    # 准备绘图数据
    wave_data = {
        'final_waves': final_waves,
        'merged_waves': merged_waves
    }
    
    # 绘制概览图
    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()
    
    print("正在生成概览图表...")
    plot_overview_chart(
        close_prices, 
        wave_data,
        start_date_all, 
        end_date_all, 
        rsi_drop_threshold, 
        rsi_rise_ratio
    )
    
    # 绘制每个波浪的详细图
    all_processed_waves = final_waves + merged_waves
    if all_processed_waves:
        print("\n正在为每个识别出的波浪生成独立的放大图...")
        for i, wave in enumerate(all_processed_waves):
            wave_indices = wave['indices']
            wave_type = wave['type']
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=15, wave_number=i+1, wave_type=wave_type)
    
    print("\n=== 波浪检测完成 ===")
    total_waves = len(final_waves) + len(merged_waves)
    print(f"总共识别出 {total_waves} 个有效波浪结构")
    
    # 返回分析结果
    return {
        'close_prices': close_prices,
        'rsi_series': rsi_series,
        'extremas': extremas,
        'trigger_points': all_trigger_points,
        'final_waves': final_waves,
        'merged_waves': merged_waves,
        'total_waves': total_waves
    }

def main():
    """
    主函数
    """
    # 配置参数
    file_path = 'BTC.csv'
    rsi_period = 14
    rsi_drop_threshold = 10
    rsi_rise_ratio = 1/3
    
    # 执行分析
    results = run_wave_analysis(
        file_path=file_path,
        rsi_period=rsi_period,
        rsi_drop_threshold=rsi_drop_threshold,
        rsi_rise_ratio=rsi_rise_ratio
    )
    
    return results

if __name__ == "__main__":
    main()