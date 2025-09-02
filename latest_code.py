import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 所有波浪识别和绘图函数 (与之前代码相同)
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

# === MODIFIED FUNCTION: is_downtrend_seven_wave ===
# 检查一个8个点的波浪是否符合下跌7浪结构，并加入P8 < P6的条件
def is_downtrend_seven_wave(prices):
    if len(prices) != 8:
        return False
    p0, p1, p2, p3, p4, p5, p6, p7 = prices.values
    # 检查交替高低点
    if not (p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5 and p5 < p6 and p6 > p7):
        return False
        
    # 添加新条件: P8 必须低于 P6
    if not (p7 < p5):
        return False
    
    # 所有条件都满足，返回True
    return True

def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(com=period - 1, adjust=False).mean()
    avg_loss = losses.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# === MODIFIED FUNCTION: find_downtrend_wave_patterns ===
# 泛化波浪寻找函数，可以寻找指定长度的波浪
def find_downtrend_wave_patterns(close_prices, extremas, length=6):
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

# === NEW MASTER FUNCTION: handle_overlapping_waves ===
def handle_overlapping_waves(all_waves, close_prices, all_extremas):
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



# === MODIFIED FUNCTION: plot_individual_wave ===
# Added if_plot_rsi parameter to conditionally plot the RSI subplot
def plot_individual_wave(file_path, close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=15, wave_number=1, wave_type="", if_plot_rsi=True):
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    
    # Conditionally create subplots
    if if_plot_rsi:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(plot_prices.index, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax2 = None # Set ax2 to None so it can be checked later
        ax1.plot(plot_prices.index, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'magenta'
    ax1.plot(wave_points_dates, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    # Dynamic point annotations remain the same
    for j in range(len(wave_indices)):
        ax1.annotate(f'P{j+1}', (wave_points_dates[j], wave_points_prices[j]), xytext=(5, 5), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
        
    for point in trigger_points:
        if point['date'] >= start_date and point['date'] <= end_date:
            trigger_price = close_prices.loc[point['date']]
            ax1.axvline(point['date'], color='gray', linestyle='--', alpha=0.5)
            marker = 'v' if point['type'] == 'drop' else '^'
            label = 'RSI Drop Trigger' if point['type'] == 'drop' else 'RSI Rise Trigger'
            ax1.plot(point['date'], trigger_price, marker, color='orange', markersize=8, label=label)
    
    ax1.set_title(f'{file_path[:4]} {wave_type.capitalize()} Wave {wave_number}: {wave_start_date.date()} to {wave_end_date.date()}', fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax1.legend(unique_labels.values(), unique_labels.keys())
    ax1.grid(True)
    
    # Only plot the RSI subplot if if_plot_rsi is True
    if if_plot_rsi:
        plot_rsi = rsi_series.loc[start_date:end_date]
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

# =========================================================
# find_extremas_with_rsi 函数 (保持不变)
# =========================================================
def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3):
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
                    'peak_rsi': peak_rsi,
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
                    'valley_rsi': valley_rsi,
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

def main(file_path='BTC.csv', rsi_drop_threshold = 15, rsi_rise_ratio = 1/4):

    # ... (all functions remain the same as the previous response) ...

    # =========================================================
    # 主程序
    # =========================================================
    #file_path = 'ETH.csv'
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

    #rsi_drop_threshold = 15
    #rsi_rise_ratio = 1/4

    extremas, all_trigger_points = find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=rsi_drop_threshold, rsi_rise_ratio=rsi_rise_ratio)

    print("正在从 RSI 驱动的峰谷点中寻找波浪结构...")
    all_waves = find_downtrend_wave_patterns(close_prices, extremas, length=6)

    final_waves, merged_waves = handle_overlapping_waves(all_waves, close_prices, extremas)

    try:
        plt.style.use('default')
    except:
        pass
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()

    # Subplot 1: Price Chart
    ax1.plot(close_prices.index, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)

    plotted_merged_label = False
    plotted_strict_label = False
    plotted_relaxed_label = False

    if merged_waves:
        print(f"\n成功识别出 {len(merged_waves)} 个合并波浪结构。")
        for i, wave in enumerate(merged_waves):
            wave_indices = wave['indices']
            wave_points_dates = close_prices.index[wave_indices]
            label = 'Merged Wave' if not plotted_merged_label else None
            ax1.plot(wave_points_dates, close_prices.iloc[wave_indices].values, 'ms-', markersize=6, label=label)
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

            ax1.plot(wave_points_dates, close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
            
            print(f"   - {wave_type.capitalize()} 波浪 {i+1}: 从 {wave_points_dates[0].date()} 到 {wave_points_dates[-1].date()}。")
            ax1.annotate(f'{wave_type.capitalize()} Wave {i+1}', (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), xytext=(5, 10), textcoords='offset points', fontsize=10, color=color, fontweight='bold')
            
    ax1.legend(loc='upper right')
    title_str = f'{file_path[:4]} Price Chart: {start_date_all} to {end_date_all} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)

    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index, rsi_series.values, color='purple', label='14-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)

    # --- MODIFIED SECTION ---
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

    all_processed_waves = final_waves + merged_waves
    if all_processed_waves:
        print("\n正在为每个识别出的波浪生成独立的放大图...")
        for i, wave in enumerate(all_processed_waves):
            wave_indices = wave['indices']
            wave_type = wave['type']
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            wave_trigger_points = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            plot_individual_wave(file_path,close_prices, rsi_series, wave_indices, wave_trigger_points, plot_range_days=15, wave_number=i+1, wave_type=wave_type,if_plot_rsi=True)
            
    print("\n=== 波浪检测完成 ===")

    print(f"总共识别出 {len(final_waves) + len(merged_waves)} 个有效波浪结构。")

if __name__ == "__main__":
    main(file_path='BTC.csv', rsi_drop_threshold = 10, rsi_rise_ratio = 1/3)