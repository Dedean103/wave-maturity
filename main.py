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
        lines = content.split('\n')
        for line in lines:
            if '原始识别:' in line and '个波浪结构' in line:
                # 提取数字
                import re
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
    print("=== BTC 波浪检测系统 (Enhanced RSI P0 Selection) ===")
    print("基于最高RSI点起始的连续波浪检测系统")
    print("支持智能P0选择、重叠波浪合并和基准质量验证")
    print()
    
    print(f"配置参数:")
    print(f"  - 数据文件: {file_path}")
    print(f"  - 回望天数 (P0搜索): {recent_days}")
    print(f"  - 最小间隔要求: {recent_days}")
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
    
    # 使用新的基于高RSI点的连续波浪检测逻辑
    print("正在执行基于最高RSI点的连续波浪检测...")
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
    try:
        plot_overview_chart(
            close_prices, 
            wave_data,
            start_date_all, 
            end_date_all, 
            rsi_drop_threshold, 
            rsi_rise_ratio,
            recent_days,
            rsi_period,
            0.95,  # trend_threshold
            price_refinement_window
        )
    except Exception as e:
        print(f"绘图出错: {e}")
        print("跳过绘图，继续分析...")
    
    # 绘制每个波浪的详细图
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
                plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=30, wave_number=i+1, wave_type=wave_type)
            except Exception as e:
                print(f"绘制波浪 {i+1} 时出错: {e}")
                continue
    
    print("\n=== 波浪检测完成 ===")
    total_waves = len(final_waves) + len(merged_waves)
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
        elif detection_rate >= 0.8:
            print("✅ 检测率满足质量要求 (≥80%)")
    
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