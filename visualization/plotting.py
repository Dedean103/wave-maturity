import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type=""):
    """
    绘制单个波浪的详细图表，包括价格和 RSI
    
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
    
    # 上半部分：价格图
    ax1.plot(plot_prices.index, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'purple'
    ax1.plot(wave_points_dates, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    for j in range(6):
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
    
    # 下半部分：RSI图
    ax2.plot(plot_rsi.index, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
    wave_points_rsi = rsi_series.loc[wave_points_dates].values
    ax2.plot(wave_points_dates, wave_points_rsi, 'o', color=color, markersize=6)
    for j in range(6):
        ax2.annotate(f'P{j+1} ({wave_points_rsi[j]:.2f})', (wave_points_dates[j], wave_points_rsi[j]), xytext=(5, 5), textcoords='offset points', fontsize=8, color=color, fontweight='bold')
    
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            marker = 'v' if point['type'] == 'drop' else '^'
            ax2.plot(point['date'], point['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            ax2.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)

            # RSI 值的注释
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
    
    # 创建文本框显示 RSI 触发事件
    textbox_content = "RSI Trigger Events:\n"
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            event_type = "Drop Trigger" if point['type'] == 'drop' else "Rise Trigger"
            trigger_date = point['date'].strftime('%Y-%m-%d')
            trigger_rsi = point['rsi_value']
            threshold = point.get('threshold', 'N/A')
            change = point.get('change', 'N/A')
            
            message = ""
            if point['type'] == 'drop':
                prev_extreme_rsi = point.get('peak_rsi', 'N/A')
                message = f"[{trigger_date}] {event_type}. From Peak {prev_extreme_rsi:.2f}, dropped to {trigger_rsi:.2f}. Change: {change:.2f} > Threshold: {threshold:.2f}."
            else:  # rise trigger
                prev_extreme_rsi = point.get('valley_rsi', 'N/A')
                message = f"[{trigger_date}] {event_type}. From Valley {prev_extreme_rsi:.2f}, rose to {trigger_rsi:.2f}. Change: {change:.2f} > Threshold: {threshold:.2f}."
            
            textbox_content += f"- {message}\n"
    
    ax2.text(1, 1, textbox_content, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

def plot_overview_chart(close_prices, strict_waves, relaxed_waves, start_date_all, end_date_all, rsi_drop_threshold, rsi_rise_ratio):
    """
    绘制概览图表，显示所有识别的波浪
    
    参数:
        close_prices (pd.Series): 收盘价序列
        strict_waves (list): 严格波浪列表
        relaxed_waves (list): 宽松波浪列表
        start_date_all (datetime): 开始日期
        end_date_all (datetime): 结束日期
        rsi_drop_threshold (float): RSI 下跌阈值
        rsi_rise_ratio (float): RSI 上涨比例
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.plot(close_prices.index, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)

    if strict_waves:
        print(f"\n成功识别出 {len(strict_waves)} 个不重叠的严格下跌五浪结构。")
        for i, wave_indices in enumerate(strict_waves):
            wave_points_dates = close_prices.index[wave_indices]
            ax.plot(wave_points_dates, close_prices.iloc[wave_indices].values, 'ro-', markersize=6)
            print(f"  - 结构 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax.annotate(f'Strict Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color='red', fontweight='bold')

    if relaxed_waves:
        print(f"\n成功识别出 {len(relaxed_waves)} 个不重叠的宽松下跌五浪结构。")
        for i, wave_indices in enumerate(relaxed_waves):
            wave_points_dates = close_prices.index[wave_indices]
            ax.plot(wave_points_dates, close_prices.iloc[wave_indices].values, 'go-', markersize=6)
            print(f"  - 结构 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax.annotate(f'Relaxed Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color='green', fontweight='bold')
            
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = sorted(set(labels), key=labels.index)
    ax.legend([handles[labels.index(l)] for l in unique_labels], unique_labels, loc='upper right')
    ax.set_title(f'BTC Price Chart: {start_date_all} to {end_date_all} (All Waves Marked), rsi_drop_threshold={rsi_drop_threshold}, rsi_rise_ratio={round(rsi_rise_ratio,2)}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    plt.tight_layout()
    plt.show()
