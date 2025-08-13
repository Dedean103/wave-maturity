import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def find_all_wave_patterns(close_prices, extremas):
    strict_waves = []
    relaxed_waves = []
    extremas_indices = [close_prices.index.get_loc(idx) for idx in extremas]
    
    for i in range(len(extremas_indices) - 5):
        points_indices = extremas_indices[i:i+6]
        points_prices = close_prices.iloc[points_indices]
        
        is_alternating = True
        for j in range(len(points_prices) - 1):
            # 修正：判断逻辑符合下跌五浪 P1>P2<P3>P4<P5>P6
            if j % 2 == 0 and points_prices.iloc[j] <= points_prices.iloc[j+1]:
                is_alternating = False
                break
            elif j % 2 != 0 and points_prices.iloc[j] >= points_prices.iloc[j+1]:
                is_alternating = False
                break
        
        if is_alternating:
            if is_downtrend_five_wave_strict(points_prices): strict_waves.append(points_indices)
            elif is_downtrend_five_wave_relaxed(points_prices): relaxed_waves.append(points_indices)
            
    return strict_waves, relaxed_waves

def remove_overlapping_waves(waves_list, close_prices):
    if not waves_list: return []
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
                is_overlapping = True; break
        if not is_overlapping:
            non_overlapping_waves.append(wave)
    return [w['indices'] for w in non_overlapping_waves]



"""
def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type=""):
    # 修正: 移除 .index() 后的括号
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    # 修正: 移除 .loc() 后的括号
    plot_prices = close_prices.loc[start_date:end_date]
    plot_rsi = rsi_series.loc[start_date:end_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # 上半部分：价格图
    ax1.plot(plot_prices.index, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    # 修正: 移除 .index() 后的括号
    wave_points_dates = close_prices.index[wave_indices]
    
    # 修正: 移除 .iloc() 后的括号
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'purple'
    ax1.plot(wave_points_dates, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    for j in range(6):
        # 修正: 移除 .iloc() 后的括号
        ax1.annotate(f'P{j+1}', (wave_points_dates[j], wave_points_prices[j]), xytext=(5, 5), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
        
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            # 修正: 移除 .loc() 后的括号
            trigger_price = close_prices.loc[point['date']]
            ax1.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            marker = 'v' if point['type'] == 'drop' else '^'
            label = 'RSI Drop Trigger' if point['type'] == 'drop' else 'RSI Rise Trigger'
            # 修正: 修复图例重复的问题
            ax1.plot(point['date'], trigger_price, marker, color='orange', markersize=8, label=label)
            
    ax1.set_title(f'BTC {wave_type.capitalize()} Wave {wave_number}: {wave_start_date.date()} to {wave_end_date.date()}', fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    # 修正: 修复图例重复的问题
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys())
    ax1.grid(True)
    
    # 下半部分：RSI图
    ax2.plot(plot_rsi.index, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
    # 修正: 移除 .loc() 和 .iloc() 后的括号
    wave_points_rsi = rsi_series.loc[wave_points_dates].values
    ax2.plot(wave_points_dates, wave_points_rsi, 'o', color=color, markersize=6)
    for j in range(6):
        # 修正: 移除 .iloc() 后的括号
        ax2.annotate(f'P{j+1} ({wave_points_rsi[j]:.2f})', (wave_points_dates[j], wave_points_rsi[j]), xytext=(5, 5), textcoords='offset points', fontsize=8, color=color, fontweight='bold')
    
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            marker = 'v' if point['type'] == 'drop' else '^'
            # 修正: 修复图例重复的问题
            ax2.plot(point['date'], point['rsi_value'], marker, color='orange', markersize=8, label='Trigger Point')
            ax2.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            # 添加 RSI 值的注释
            rsi_value_text = f"{point['rsi_value']:.2f}"
            y_offset = -10 if point['type'] == 'drop' else 10
            ax2.annotate(rsi_value_text, (point['date'], point['rsi_value']),
                         xytext=(0, y_offset), textcoords='offset points',
                         ha='center', va='center', fontsize=8, color='orange',
                         bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))

    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    # 修正: 修复图例重复的问题
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    
    plt.tight_layout()
    plt.show()


# =========================================================
# 第二部分：修正实时峰谷识别函数，使用完整波段RSI变化
# =========================================================
def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=15, rsi_rise_ratio=1/3):
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    extremas = []
    trigger_points = []
    
    print("\n--- 开始 RSI 驱动的峰谷点识别 ---")
    
    if len(rsi_series) < 2: return extremas, trigger_points
    
    # 初始化
    peak_rsi = rsi_series.iloc[0]
    valley_rsi = rsi_series.iloc[0]
    peak_date = rsi_series.index[0]
    valley_date = rsi_series.index[0]
    
    # 追踪上一个完整波段的极值点，用于动态阈值
    last_wave_peak_rsi = peak_rsi
    
    direction = 0 # 0: 初始状态, 1: 上升趋势, -1: 下降趋势
    
    for i in range(1, len(rsi_series)):
        current_rsi = rsi_series.iloc[i]
        current_date = rsi_series.index[i]
        
        if direction >= 0: # 寻找波峰
            if current_rsi > peak_rsi:
                peak_rsi = current_rsi
                peak_date = current_date
            
            rsi_drop_change = peak_rsi - current_rsi
            if rsi_drop_change >= rsi_drop_threshold:
                if not extremas or extremas[-1] != peak_date:
                    extremas.append(peak_date)
                
                trigger_points.append({'date': current_date, 'type': 'drop', 'rsi_value': current_rsi})
                
                print(f"[{current_date.date()}] 触发 RSI Drop Trigger!")
                print(f"   -> 确认波峰在: {peak_date.date()}")
                print(f"   -> RSI 从最高点 {peak_rsi:.2f} 跌至 {current_rsi:.2f}，累计下跌 {rsi_drop_change:.2f} 点，满足阈值 {rsi_drop_threshold}。")
                
                # 状态切换：从上升到下降
                direction = -1
                last_wave_peak_rsi = peak_rsi
                valley_rsi = current_rsi
                valley_date = current_date
        
        if direction <= 0: # 寻找波谷
            if current_rsi < valley_rsi:
                valley_rsi = current_rsi
                valley_date = current_date
            
            rsi_rise_change = current_rsi - valley_rsi
            # 动态阈值计算
            drop_amount_for_threshold = last_wave_peak_rsi - valley_rsi
            rise_threshold = drop_amount_for_threshold * rsi_rise_ratio
            
            if rsi_rise_change >= rise_threshold and drop_amount_for_threshold > 0:
                if not extremas or extremas[-1] != valley_date:
                    extremas.append(valley_date)
                
                trigger_points.append({'date': current_date, 'type': 'rise', 'rsi_value': current_rsi})
                
                print(f"[{current_date.date()}] 触发 RSI Rise Trigger!")
                print(f"   -> 确认波谷在: {valley_date.date()}")
                print(f"   -> RSI 从最低点 {valley_rsi:.2f} 涨至 {current_rsi:.2f}，累计上涨 {rsi_rise_change:.2f} 点，满足阈值 {rise_threshold:.2f}。")
                
                # 状态切换：从下降到上升
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    print("\n--- RSI 驱动的峰谷点识别结束 ---")
    return extremas, trigger_points


def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type=""):
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    plot_rsi = rsi_series.loc[start_date:end_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Upper subplot: Price Chart
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
    
    # Lower subplot: RSI Chart
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
            
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    
    # ====================================================
    # 新增逻辑：在RSI图内创建文本框
    # ====================================================
    textbox_content = "RSI Trigger Events:\n"
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            event_type = "Drop Trigger" if point['type'] == 'drop' else "Rise Trigger"
            
            trigger_date = point['date'].strftime('%Y-%m-%d')
            trigger_rsi = point['rsi_value']
            
            prev_extreme_rsi = None
            if point['type'] == 'drop':
                prev_extreme_rsi = point.get('peak_rsi', 'N/A')
                message = f"[{trigger_date}] {event_type}. Peak RSI: {prev_extreme_rsi:.2f}, drops to {trigger_rsi:.2f}."
            else: # rise trigger
                prev_extreme_rsi = point.get('valley_rsi', 'N/A')
                message = f"[{trigger_date}] {event_type}. Valley RSI: {prev_extreme_rsi:.2f}, rises to {trigger_rsi:.2f}."
            
            textbox_content += f"- {message}\n"
    
    # 使用 ax2.text() 在 RSI 图的右上角添加文本框
    ax2.text(1, 1, textbox_content, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

"""

def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type=""):
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    plot_rsi = rsi_series.loc[start_date:end_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Upper subplot: Price Chart
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
    
    # Lower subplot: RSI Chart
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

            # --- RESTORED LOGIC for individual RSI value labels ---
            rsi_value_text = f"{point['rsi_value']:.2f}"
            y_offset = -10 if point['type'] == 'drop' else 10
            ax2.annotate(rsi_value_text, (point['date'], point['rsi_value']),
                         xytext=(0, y_offset), textcoords='offset points',
                         ha='center', va='center', fontsize=10, color='orange',
                         bbox=dict(facecolor='white', edgecolor='lightgray', boxstyle='round,pad=0.3'))
            # ----------------------------------------------------
            
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    handles, labels = ax2.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax2.legend(unique_labels.values(), unique_labels.keys())
    
    # ====================================================
    # Logic for textbox is still here
    # ====================================================
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
            else: # rise trigger
                prev_extreme_rsi = point.get('valley_rsi', 'N/A')
                message = f"[{trigger_date}] {event_type}. From Valley {prev_extreme_rsi:.2f}, rose to {trigger_rsi:.2f}. Change: {change:.2f} > Threshold: {threshold:.2f}."
            
            textbox_content += f"- {message}\n"
    
    ax2.text(1, 1, textbox_content, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.show()

# =========================================================
# 修正后的 find_extremas_with_rsi 函数 (保留不变，但确保保存了峰谷RSI)
# =========================================================
def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=15, rsi_rise_ratio=1/3):
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    extremas = []
    trigger_points = []
    
    print("\n--- Starting RSI-driven Extremas Identification ---")
    
    if len(rsi_series) < 2: return extremas, trigger_points
    
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
                if not extremas or extremas[-1] != peak_date:
                    extremas.append(peak_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'drop', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,  # 保存波峰RSI
                    'valley_rsi': valley_rsi,
                    'change': rsi_drop_change,
                    'threshold': rsi_drop_threshold
                })
                
                print(f"[{current_date.date()}] Triggered RSI Drop Trigger!")
                print(f"   -> Confirmed Peak at: {peak_date.date()}")
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
            
            if rsi_rise_change >= rise_threshold and drop_amount_for_threshold > 0:
                if not extremas or extremas[-1] != valley_date:
                    extremas.append(valley_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'rise', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,  # 保存波谷RSI
                    'change': rsi_rise_change,
                    'threshold': rise_threshold
                })
                
                print(f"[{current_date.date()}] Triggered RSI Rise Trigger!")
                print(f"   -> Confirmed Valley at: {valley_date.date()}")
                print(f"   -> RSI rose from low {valley_rsi:.2f} to {current_rsi:.2f}, total rise {rsi_rise_change:.2f} points, meeting threshold {rise_threshold:.2f}.")
                
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    print("\n--- RSI-driven Extremas Identification Finished ---")
    return extremas, trigger_points



file_path = 'btc.csv'
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。请确保文件存在并位于正确的路径下。")
    exit()
    
if 'datetime' not in data.columns or 'close' not in data.columns:
    print("错误: CSV文件必须包含 'datetime' 和 'close' 两列。")
    exit()
    
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)
close_prices = data['close']

rsi_series = calculate_rsi(close_prices)
rsi_drop_threshold= 10
rsi_rise_ratio= 1/4
extremas, all_trigger_points = find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=rsi_drop_threshold, rsi_rise_ratio=rsi_rise_ratio)
print("正在从 RSI 驱动的峰谷点中寻找波浪结构...")
strict_waves_all, relaxed_waves_all = find_all_wave_patterns(close_prices, extremas)

strict_waves = remove_overlapping_waves(strict_waves_all, close_prices)
relaxed_waves = remove_overlapping_waves(relaxed_waves_all, close_prices)

plt.style.use('seaborn-v0_8-darkgrid')
fig, ax = plt.subplots(figsize=(15, 8))
start_date_all = close_prices.index.min().date()
end_date_all = close_prices.index.max().date()
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

if strict_waves:
    print("\n正在为每个识别出的严格波浪生成独立的放大图...")
    for i, wave_indices in enumerate(strict_waves):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=15, wave_number=i+1, wave_type='strict')

if relaxed_waves:
    print("\n正在为每个识别出的宽松波浪生成独立的放大图...")
    for i, wave_indices in enumerate(relaxed_waves):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=15, wave_number=i+1, wave_type='relaxed')