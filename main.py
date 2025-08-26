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
from datetime import datetime

# 添加项目根目录到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.wave_detection import find_downtrend_wave_patterns, handle_overlapping_waves, find_continuous_waves_from_recent_highs
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

def run_wave_analysis(file_path='BTC.csv', recent_days=50, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3, price_refinement_window=5):
    """
    执行完整的波浪分析流程 - 更新版本，使用最近高RSI点作为起始点
    
    参数:
        file_path (str): 数据文件路径
        recent_days (int): 搜索最近天数和最小间隔天数
        rsi_period (int): RSI 计算周期
        rsi_drop_threshold (int): RSI 下跌触发阈值
        rsi_rise_ratio (float): RSI 上涨触发比例
        price_refinement_window (int): 价格微调窗口天数
    
    返回:
        dict: 分析结果字典
    """
    print("=== BTC 波浪检测系统 (Latest Version with Recent High RSI Start) ===")
    print("基于最近高RSI点的持续波浪检测系统")
    print("支持智能P0选择、重叠波浪合并和持续波浪发现")
    print()
    
    print(f"配置参数:")
    print(f"  - 数据文件: {file_path}")
    print(f"  - 搜索最近天数: {recent_days}")
    print(f"  - RSI 周期: {rsi_period}")
    print(f"  - RSI 下跌阈值: {rsi_drop_threshold}")
    print(f"  - RSI 上涨比例: {rsi_rise_ratio}")
    print(f"  - 价格微调窗口: {price_refinement_window}")
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
    
    # 使用新的持续波浪检测逻辑
    print("正在执行基于最近高RSI点的持续波浪检测...")
    all_waves, all_trigger_points = find_continuous_waves_from_recent_highs(
        close_prices, 
        recent_days=recent_days,
        rsi_period=rsi_period, 
        rsi_drop_threshold=rsi_drop_threshold, 
        rsi_rise_ratio=rsi_rise_ratio,
        price_refinement_window=price_refinement_window
    )
    
    # 处理重叠波浪（如果存在）
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
        all_extremas = []
        print("未检测到有效波浪结构")
    
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
        'extremas': all_extremas,
        'trigger_points': all_trigger_points,
        'final_waves': final_waves,
        'merged_waves': merged_waves,
        'total_waves': total_waves
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
    recent_days = 50
    rsi_period = 14
    rsi_drop_threshold = 10
    rsi_rise_ratio = 1/3
    price_refinement_window = 5
    
    # 打开日志文件
    with open(log_filename, 'w', encoding='utf-8') as log_file:
        # 创建同时输出到控制台和文件的对象
        original_stdout = sys.stdout
        sys.stdout = TeeOutput(original_stdout, log_file)
        
        try:
            # 执行分析
            results = run_wave_analysis(
                file_path=file_path,
                recent_days=recent_days,
                rsi_period=rsi_period,
                rsi_drop_threshold=rsi_drop_threshold,
                rsi_rise_ratio=rsi_rise_ratio,
                price_refinement_window=price_refinement_window
            )
            
            print(f"\n分析结果已保存到: {log_filename}")
            
        finally:
            # 恢复原始stdout
            sys.stdout = original_stdout
    
    print(f"分析完成，日志已保存到: {log_filename}")
    return results

if __name__ == "__main__":
    main()