import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Core wave validation functions (from latest_code.py)
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

def calculate_rsi(series, period=14):
    delta = series.diff().dropna()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.ewm(com=period - 1, adjust=False).mean()
    avg_loss = losses.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# =========================================================
# Trend validation functions
# =========================================================

def validate_wave_overall_trend(close_prices, wave_indices, trend_threshold=0.95):
    """
    Validate that a wave occurs within an overall downtrend
    """
    start_idx = wave_indices[0]  # P1
    end_idx = wave_indices[-1]   # P6
    
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
# RSI-driven extrema detection (from latest_code.py)
# =========================================================

def find_extremas_with_rsi(close_prices, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3):
    rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    extremas = []
    trigger_points = []
    
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
                
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    return extremas, trigger_points

# =========================================================
# Pattern validation with dual filtering
# =========================================================

def find_valid_wave_patterns(close_prices, extremas, trend_threshold=0.95):
    """
    Find downtrend wave patterns with dual filtering (pattern + trend)
    """
    valid_waves = []
    extremas_indices = [close_prices.index.get_loc(idx) for idx in extremas]
    
    # Try to find 6-point waves
    for i in range(len(extremas_indices) - 6 + 1):
        points_indices = extremas_indices[i:i+6]
        points_prices = close_prices.iloc[points_indices]
        
        # Pattern validation
        wave_type = None
        if is_downtrend_five_wave_strict(points_prices): 
            wave_type = 'strict'
        elif is_downtrend_five_wave_relaxed(points_prices): 
            wave_type = 'relaxed'
        
        # If pattern is valid, check trend validation
        if wave_type:
            is_trend_valid, trend_ratio, trend_info = validate_wave_overall_trend(
                close_prices, points_indices, trend_threshold
            )
            
            if is_trend_valid:
                valid_waves.append({
                    'indices': points_indices, 
                    'type': wave_type,
                    'trend_ratio': trend_ratio,
                    'trend_info': trend_info
                })
                print(f"✓ Valid {wave_type} wave: {trend_info['start_date']} to {trend_info['end_date']}, trend ratio: {trend_ratio:.3f}")
                return valid_waves  # Return first valid wave found
            
    return valid_waves

# =========================================================
# Sequential RSI P0 + Progressive Wave Detection (Approach 2)
# =========================================================

def sequential_wave_detection(close_prices, lookback_days=50, rsi_period=14, 
                            rsi_drop_threshold=10, rsi_rise_ratio=1/3, 
                            trend_threshold=0.95, recent_days=50):
    """
    Sequential P0 selection with progressive wave detection and dual filtering
    """
    print(f"=== Sequential RSI P0 + Progressive Wave Detection ===")
    print(f"Parameters: lookback_days={lookback_days}, trend_threshold={trend_threshold}")
    
    all_waves = []
    all_trigger_points = []
    current_position = 0
    wave_count = 0
    max_iterations = 20
    
    while current_position < len(close_prices) - lookback_days and wave_count < max_iterations:
        wave_count += 1
        print(f"\n--- Wave Search {wave_count} ---")
        
        # Define search window for P0
        end_position = min(current_position + lookback_days, len(close_prices))
        search_data = close_prices.iloc[current_position:end_position]
        
        if len(search_data) < 30:  # Need sufficient data
            print(f"Insufficient data remaining. Stopping search.")
            break
            
        print(f"P0 search window: {search_data.index[0].date()} to {search_data.index[-1].date()} ({len(search_data)} days)")
        
        # Find highest RSI point in window as P0
        search_rsi = calculate_rsi(search_data, period=rsi_period)
        if search_rsi.isna().all():
            print("No valid RSI data. Advancing position.")
            current_position += lookback_days // 2
            continue
            
        p0_date = search_rsi.idxmax()
        p0_rsi = search_rsi.max()
        print(f"P0 selected: {p0_date.date()}, RSI: {p0_rsi:.2f}")
        
        # Progressive wave detection from P0
        p0_position = close_prices.index.get_loc(p0_date)
        remaining_data = close_prices.iloc[p0_position:]
        
        if len(remaining_data) < 30:
            print("Insufficient data after P0. Advancing position.")
            current_position += lookback_days // 2
            continue
        
        # Progressive RSI extrema detection with wave search
        print("Starting progressive extrema detection and wave search...")
        extremas, triggers = find_extremas_with_rsi(
            remaining_data, rsi_period, rsi_drop_threshold, rsi_rise_ratio
        )
        
        # Ensure P0 is first extrema
        if not extremas or extremas[0] != p0_date:
            extremas.insert(0, p0_date)
        
        print(f"Found {len(extremas)} extrema points")
        
        # Try to find valid waves with dual filtering
        if len(extremas) >= 6:
            valid_waves = find_valid_wave_patterns(close_prices, extremas, trend_threshold)
            
            if valid_waves:
                wave = valid_waves[0]  # Take first valid wave
                all_waves.append(wave)
                all_trigger_points.extend(triggers)
                
                # Advance past this wave
                wave_end_idx = wave['indices'][-1]
                current_position = wave_end_idx + recent_days
                print(f"Wave found! Advancing to position {current_position}")
            else:
                print("No valid waves found (rejected by dual filtering). Advancing position.")
                current_position += lookback_days // 2
        else:
            print(f"Only {len(extremas)} extrema points. Need ≥6. Advancing position.")
            current_position += lookback_days // 2
    
    print(f"\n=== Sequential Detection Complete ===")
    print(f"Total valid waves: {len(all_waves)}")
    
    return all_waves, all_trigger_points

# =========================================================
# Plotting functions (simplified from latest_code.py)
# =========================================================

def plot_overview_chart(close_prices, all_waves, rsi_drop_threshold, rsi_rise_ratio):
    """
    Plot overview chart showing all detected waves
    """
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot price line (fix pandas indexing issue)
    ax.plot(close_prices.index.values, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    # Plot waves
    plotted_strict_label = False
    plotted_relaxed_label = False
    
    for i, wave in enumerate(all_waves):
        wave_indices = wave['indices']
        wave_type = wave['type']
        wave_points_dates = close_prices.index[wave_indices]
        color = 'red' if wave_type == 'strict' else 'green'
        
        # Only show label once per type
        if wave_type == 'strict' and not plotted_strict_label:
            label = 'Strict Wave'
            plotted_strict_label = True
        elif wave_type == 'relaxed' and not plotted_relaxed_label:
            label = 'Relaxed Wave'
            plotted_relaxed_label = True
        else:
            label = None

        ax.plot(wave_points_dates.values, close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
        ax.annotate(f'{wave_type.capitalize()} Wave {i+1}', 
                   (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), 
                   xytext=(5, 10), textcoords='offset points', 
                   fontsize=10, color=color, fontweight='bold')
        
    ax.legend(loc='upper right')
    start_date = close_prices.index.min().date()
    end_date = close_prices.index.max().date()
    title_str = f'BTC Price Chart: {start_date} to {end_date} (Valid Downtrend Waves)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}'
    ax.set_title(title_str, fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Close Price', fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def plot_individual_wave(close_prices, rsi_series, wave_indices, trigger_points, wave_number=1, wave_type="", plot_range_days=15):
    """
    Plot individual wave with RSI subplot
    """
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    plot_rsi = rsi_series.loc[start_date:end_date]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    # Price subplot (fix pandas indexing issue)
    ax1.plot(plot_prices.index.values, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green'
    ax1.plot(wave_points_dates.values, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    # Point annotations
    for j in range(len(wave_indices)):
        ax1.annotate(f'P{j+1}', (wave_points_dates[j], wave_points_prices[j]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold')
        
    ax1.set_title(f'BTC {wave_type.capitalize()} Wave {wave_number}: {wave_start_date.date()} to {wave_end_date.date()}', fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.legend()
    ax1.grid(True)
    
    # RSI subplot (fix pandas indexing issue)
    ax2.plot(plot_rsi.index.values, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
    wave_points_rsi = rsi_series.loc[wave_points_dates].values
    ax2.plot(wave_points_dates.values, wave_points_rsi, 'o', color=color, markersize=6)
    
    for j in range(len(wave_indices)):
        ax2.annotate(f'P{j+1} ({wave_points_rsi[j]:.2f})', 
                    (wave_points_dates[j], wave_points_rsi[j]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color=color, fontweight='bold')
    
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# =========================================================
# Main function
# =========================================================

def main(file_path='BTC.csv'):
    print("=== BTC Wave Detection System (Approach 2) ===")
    print("Sequential P0 + Progressive Detection with Dual Filtering")
    
    # Load data
    try:
        data = pd.read_csv(file_path)
        data['datetime'] = pd.to_datetime(data['datetime'])
        data.set_index('datetime', inplace=True)
        close_prices = data['close']
        print(f"Loaded {len(close_prices)} data points: {close_prices.index.min().date()} to {close_prices.index.max().date()}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Calculate RSI for plotting
    rsi_series = calculate_rsi(close_prices)
    
    # Run sequential wave detection
    all_waves, all_trigger_points = sequential_wave_detection(close_prices)
    
    # Results summary
    print(f"\n=== Results ===")
    print(f"Total waves detected: {len(all_waves)}")
    for i, wave in enumerate(all_waves):
        trend_info = wave['trend_info']
        print(f"Wave {i+1}: {wave['type']} - {trend_info['start_date']} to {trend_info['end_date']}")
    
    # Plot overview chart
    if all_waves:
        print("\nGenerating overview chart...")
        plot_overview_chart(close_prices, all_waves, 10, 1/3)
        
        # Plot individual waves
        print("Generating individual wave charts...")
        for i, wave in enumerate(all_waves):
            wave_indices = wave['indices']
            wave_type = wave['type']
            wave_start_date = close_prices.index[wave_indices[0]]
            wave_end_date = close_prices.index[wave_indices[-1]]
            
            # Get relevant trigger points for this wave
            wave_trigger_points = [p for p in all_trigger_points 
                                 if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
            
            plot_individual_wave(close_prices, rsi_series, wave_indices, wave_trigger_points, 
                               wave_number=i+1, wave_type=wave_type)

if __name__ == "__main__":
    main()