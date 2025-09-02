import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=30, wave_number=1, wave_type=""):
    """
    绘制单个波浪的详细图表，包括价格和 RSI
    支持6点和8点波浪的动态显示
    
    参数:
        close_prices (pd.Series): 收盘价序列
        rsi_series (pd.Series): RSI 序列
        wave_indices (list): 波浪点索引
        trigger_points (list): 触发点列表
        plot_range_days (int): 绘图范围天数
        wave_number (int): 波浪编号
        wave_type (str): 波浪类型
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
    
    # 动态处理点位标注，以适应6点和8点波浪
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

def plot_overview_chart(close_prices, all_waves, start_date_all, end_date_all, rsi_drop_threshold, rsi_rise_ratio, 
                        lookback_days=50, rsi_period=14, trend_threshold=0.95, price_refinement_window=5):
    """
    绘制概览图表，显示所有识别的波浪（支持新的波浪结构）
    
    参数:
        close_prices (pd.Series): 收盘价序列
        all_waves (dict): 包含final_waves和merged_waves的字典
        start_date_all (datetime): 开始日期
        end_date_all (datetime): 结束日期
        rsi_drop_threshold (float): RSI 下跌阈值
        rsi_rise_ratio (float): RSI 上涨比例
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
    
    # Add parameter textbox
    param_text = f"""Parameters:
• Lookback Days: {lookback_days}
• RSI Period: {rsi_period}
• RSI Drop Threshold: {rsi_drop_threshold}
• RSI Rise Ratio: {rsi_rise_ratio:.3f}
• Trend Threshold: {trend_threshold}
• Price Refinement Window: {price_refinement_window}"""
    
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.show()