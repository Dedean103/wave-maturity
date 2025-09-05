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
# Bidirectional price refinement functions
# =========================================================

def refine_peak_with_price(close_prices, rsi_peak_date, window_days=5):
    """
    Refine RSI-confirmed peak using price data (bidirectional search)
    Look for higher price within window centered around RSI peak
    """
    try:
        peak_idx = close_prices.index.get_loc(rsi_peak_date)
        
        # Define bidirectional search window
        start_idx = max(0, peak_idx - window_days//2)
        end_idx = min(len(close_prices), peak_idx + window_days//2 + 1)
        search_window = close_prices.iloc[start_idx:end_idx]
        
        # Find highest price point in window
        max_price_idx_local = search_window.idxmax()
        max_price = search_window.max()
        
        # Get original RSI peak price for comparison
        original_price = close_prices.iloc[peak_idx]
        
        # If higher price found, adjust peak point
        if max_price > original_price:
            print(f"   -> Price refinement: Peak from {rsi_peak_date.date()} (${original_price:.2f}) to {max_price_idx_local.date()} (${max_price:.2f})")
            return max_price_idx_local
        else:
            return rsi_peak_date
    except:
        return rsi_peak_date

def refine_valley_with_price(close_prices, rsi_valley_date, window_days=5):
    """
    Refine RSI-confirmed valley using price data (bidirectional search)
    Look for lower price within window centered around RSI valley
    """
    try:
        valley_idx = close_prices.index.get_loc(rsi_valley_date)
        
        # Define bidirectional search window
        start_idx = max(0, valley_idx - window_days//2)
        end_idx = min(len(close_prices), valley_idx + window_days//2 + 1)
        search_window = close_prices.iloc[start_idx:end_idx]
        
        # Find lowest price point in window
        min_price_idx_local = search_window.idxmin()
        min_price = search_window.min()
        
        # Get original RSI valley price for comparison
        original_price = close_prices.iloc[valley_idx]
        
        # If lower price found, adjust valley point
        if min_price < original_price:
            print(f"   -> Price refinement: Valley from {rsi_valley_date.date()} (${original_price:.2f}) to {min_price_idx_local.date()} (${min_price:.2f})")
            return min_price_idx_local
        else:
            return rsi_valley_date
    except:
        return rsi_valley_date

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

def find_extremas_with_rsi(close_prices, rsi_series=None, rsi_period=14, rsi_drop_threshold=10, rsi_rise_ratio=1/3, price_refinement_window=5):
    # Use provided RSI series or calculate if not provided (backward compatibility)
    if rsi_series is None:
        rsi_series = calculate_rsi(close_prices, period=rsi_period).dropna()
    else:
        # Filter RSI series to match the close_prices date range
        rsi_series = rsi_series.loc[close_prices.index[0]:close_prices.index[-1]]
    
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
                # Store RSI-detected peak (no price refinement here)
                if not extremas or extremas[-1] != peak_date:
                    extremas.append(peak_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'drop', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,
                    'change': rsi_drop_change,
                    'threshold': rsi_drop_threshold,
                    'refined_peak_date': peak_date
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
                # Store RSI-detected valley (no price refinement here)
                if not extremas or extremas[-1] != valley_date:
                    extremas.append(valley_date)
                
                trigger_points.append({
                    'date': current_date, 
                    'type': 'rise', 
                    'rsi_value': current_rsi,
                    'peak_rsi': peak_rsi,
                    'valley_rsi': valley_rsi,
                    'change': rsi_rise_change,
                    'threshold': rise_threshold,
                    'refined_valley_date': valley_date
                })
                
                direction = 1
                peak_rsi = current_rsi
                peak_date = current_date
                
    if extremas and extremas[-1] not in [peak_date, valley_date]:
        extremas.append(close_prices.index[close_prices.index.get_loc(rsi_series.index[-1])])
    
    return extremas, trigger_points

# =========================================================
# Sequential wave building (NEW APPROACH)
# =========================================================

def find_next_valley_after(close_prices, rsi_series, start_date, rsi_drop_threshold, rsi_rise_ratio, price_refinement_window=5):
    """Find the next RSI-confirmed valley after start_date"""
    start_idx = close_prices.index.get_loc(start_date)
    remaining_data = close_prices.iloc[start_idx + 1:]  # Start after current point
    remaining_rsi = rsi_series.iloc[start_idx + 1:]
    
    if len(remaining_rsi) < 10:  # Need sufficient data
        return None
    
    # Track peak to find drop
    peak_rsi = remaining_rsi.iloc[0]
    peak_date = remaining_rsi.index[0]
    
    for i in range(1, len(remaining_rsi)):
        current_rsi = remaining_rsi.iloc[i]
        current_date = remaining_rsi.index[i]
        
        # Update peak
        if current_rsi > peak_rsi:
            peak_rsi = current_rsi
            peak_date = current_date
        
        # Check for RSI drop trigger
        rsi_drop = peak_rsi - current_rsi
        if rsi_drop >= rsi_drop_threshold:
            # Look for subsequent valley
            valley_rsi = current_rsi
            valley_date = current_date
            
            # Continue to find the actual valley point
            for j in range(i + 1, min(i + 20, len(remaining_rsi))):  # Look ahead for valley
                check_rsi = remaining_rsi.iloc[j]
                check_date = remaining_rsi.index[j]
                
                if check_rsi < valley_rsi:
                    valley_rsi = check_rsi
                    valley_date = check_date
                
                # Check for rise trigger to confirm valley
                rise_amount = check_rsi - valley_rsi
                drop_for_threshold = peak_rsi - valley_rsi
                rise_threshold = drop_for_threshold * rsi_rise_ratio
                
                if rise_amount >= rise_threshold and drop_for_threshold > 0:
                    # Apply price refinement
                    refined_valley = refine_valley_with_price(close_prices, valley_date, price_refinement_window)
                    return refined_valley
                    
    return None

def find_next_peak_after(close_prices, rsi_series, start_date, rsi_drop_threshold, rsi_rise_ratio, price_refinement_window=5):
    """Find the next RSI-confirmed peak after start_date"""
    start_idx = close_prices.index.get_loc(start_date)
    remaining_data = close_prices.iloc[start_idx + 1:]  # Start after current point
    remaining_rsi = rsi_series.iloc[start_idx + 1:]
    
    if len(remaining_rsi) < 10:  # Need sufficient data
        return None
    
    # Track valley to find rise
    valley_rsi = remaining_rsi.iloc[0]
    valley_date = remaining_rsi.index[0]
    
    for i in range(1, len(remaining_rsi)):
        current_rsi = remaining_rsi.iloc[i]
        current_date = remaining_rsi.index[i]
        
        # Update valley
        if current_rsi < valley_rsi:
            valley_rsi = current_rsi
            valley_date = current_date
        
        # Look for rise that could indicate a peak
        rise_from_valley = current_rsi - valley_rsi
        if rise_from_valley >= rsi_drop_threshold * rsi_rise_ratio:  # Sufficient rise
            # Look for subsequent peak
            peak_rsi = current_rsi
            peak_date = current_date
            
            # Continue to find the actual peak point  
            for j in range(i + 1, min(i + 20, len(remaining_rsi))):  # Look ahead for peak
                check_rsi = remaining_rsi.iloc[j]
                check_date = remaining_rsi.index[j]
                
                if check_rsi > peak_rsi:
                    peak_rsi = check_rsi
                    peak_date = check_date
                
                # Check for drop trigger to confirm peak
                drop_amount = peak_rsi - check_rsi
                if drop_amount >= rsi_drop_threshold:
                    # Apply price refinement
                    refined_peak = refine_peak_with_price(close_prices, peak_date, price_refinement_window)
                    return refined_peak
                    
    return None

def classify_extrema_type(close_prices, date, prev_date=None):
    """Classify if an extrema point is a peak or valley relative to previous point"""
    if prev_date is None:
        # First point - classify by looking ahead
        date_idx = close_prices.index.get_loc(date)
        if date_idx < len(close_prices) - 1:
            next_price = close_prices.iloc[date_idx + 1]
            current_price = close_prices.loc[date]
            return 'peak' if current_price > next_price else 'valley'
        return 'peak'  # Default for last point
    
    prev_price = close_prices.loc[prev_date]
    current_price = close_prices.loc[date]
    return 'peak' if current_price > prev_price else 'valley'

def build_wave_with_dynamic_correction(close_prices, p0_initial, rsi_series, rsi_drop_threshold, rsi_rise_ratio, trend_threshold=0.95, price_refinement_window=5):
    """
    Build 6-point wave with dynamic extrema correction:
    - Each new extrema either corrects previous point or becomes next point
    - P0 (peak) can be corrected by higher peaks
    - P1 (valley) can be corrected by lower valleys, etc.
    """
    
    print(f"   Building wave with dynamic correction from P0: {p0_initial.date()} (${close_prices.loc[p0_initial]:.0f})")
    
    # Initialize wave with P0
    wave_points = [p0_initial]
    expected_types = ['peak', 'valley', 'peak', 'valley', 'peak', 'valley']  # P0-P5 pattern
    current_position = 0  # Position in wave (0=P0, 1=P1, etc.)
    
    # Get remaining data after P0 for extrema detection
    p0_idx = close_prices.index.get_loc(p0_initial)
    remaining_data = close_prices.iloc[p0_idx + 1:]  # Start after P0
    
    if len(remaining_data) < 30:
        print("   Insufficient data after P0")
        return None
    
    # Find extremas in remaining data
    remaining_rsi = rsi_series.iloc[p0_idx + 1:]
    extremas, triggers = find_extremas_with_rsi(
        remaining_data, remaining_rsi, 14, rsi_drop_threshold, rsi_rise_ratio, price_refinement_window
    )
    
    print(f"   Found {len(extremas)} potential extremas after P0")
    
    # Process each extrema with dynamic correction logic
    for extrema_date in extremas:
        if len(wave_points) >= 6:  # Wave complete
            break
            
        extrema_type = classify_extrema_type(close_prices, extrema_date, wave_points[-1] if len(wave_points) > 0 else None)
        expected_type = expected_types[current_position + 1] if current_position + 1 < len(expected_types) else None
        
        print(f"   Evaluating extrema: {extrema_date.date()} (${close_prices.loc[extrema_date]:.0f}) - type: {extrema_type}, expected: {expected_type}")
        
        if extrema_type == expected_type:
            # This is the next point we're looking for - apply price refinement
            if extrema_type == 'peak':
                refined_extrema_date = refine_peak_with_price(close_prices, extrema_date, price_refinement_window)
            else:
                refined_extrema_date = refine_valley_with_price(close_prices, extrema_date, price_refinement_window)
            
            wave_points.append(refined_extrema_date)
            current_position += 1
            print(f"     -> Added as P{current_position}: {refined_extrema_date.date()}")
            
        elif current_position > 0 and extrema_type == expected_types[current_position]:
            # This could correct the current point
            current_point = wave_points[current_position]
            should_correct = False
            
            if extrema_type == 'peak':
                # Higher peak corrects current peak
                should_correct = close_prices.loc[extrema_date] > close_prices.loc[current_point]
            else:  # valley
                # Lower valley corrects current valley
                should_correct = close_prices.loc[extrema_date] < close_prices.loc[current_point]
            
            if should_correct:
                # Apply price refinement to the correction
                if extrema_type == 'peak':
                    refined_extrema_date = refine_peak_with_price(close_prices, extrema_date, price_refinement_window)
                else:
                    refined_extrema_date = refine_valley_with_price(close_prices, extrema_date, price_refinement_window)
                
                print(f"     -> Correcting P{current_position} from {current_point.date()} to {refined_extrema_date.date()}")
                wave_points[current_position] = refined_extrema_date
            else:
                print(f"     -> Not a better {extrema_type}, skipping")
        else:
            print(f"     -> Wrong type or position, skipping")
    
    # Validate complete wave
    if len(wave_points) < 6:
        print(f"   Incomplete wave: only {len(wave_points)} points found")
        return None
    
    # Take first 6 points and validate
    wave_points = wave_points[:6]
    wave_indices = [close_prices.index.get_loc(date) for date in wave_points]
    wave_prices = close_prices.iloc[wave_indices]
    
    print(f"   Final wave points: {[f'P{i}={date.date()}(${close_prices.loc[date]:.0f})' for i, date in enumerate(wave_points)]}")
    
    # Pattern validation
    wave_type = None
    if is_downtrend_five_wave_strict(wave_prices):
        wave_type = 'strict'
    elif is_downtrend_five_wave_relaxed(wave_prices):
        wave_type = 'relaxed'
    
    if not wave_type:
        print(f"   Pattern validation failed - not a valid Elliott Wave")
        # Debug: Show why pattern failed
        p0, p1, p2, p3, p4, p5 = wave_prices.values
        print(f"   Wave prices: P0=${p0:.0f} P1=${p1:.0f} P2=${p2:.0f} P3=${p3:.0f} P4=${p4:.0f} P5=${p5:.0f}")
        
        # Check basic alternating pattern
        basic_pattern = p0 > p1 and p1 < p2 and p2 > p3 and p3 < p4 and p4 > p5
        print(f"   Basic alternating pattern (P0>P1<P2>P3<P4>P5): {basic_pattern}")
        return None
    
    # Trend validation
    is_trend_valid, trend_ratio, trend_info = validate_wave_overall_trend(close_prices, wave_indices, trend_threshold)
    
    if not is_trend_valid:
        print(f"   Trend validation failed - ratio: {trend_ratio:.3f} (threshold: {trend_threshold})")
        return None
    
    print(f"   ✓ Valid {wave_type} wave found with dynamic correction! Trend ratio: {trend_ratio:.3f}")
    
    return {
        'indices': wave_indices,
        'dates': wave_points,
        'type': wave_type,
        'trend_ratio': trend_ratio,
        'trend_info': trend_info
    }

# =========================================================
# Sequential RSI P0 + Dynamic Extrema Correction (Approach 2)
# =========================================================

def sequential_wave_detection(close_prices, lookback_days=50, rsi_period=14, 
                            rsi_drop_threshold=10, rsi_rise_ratio=1/3, 
                            trend_threshold=0.95, recent_days=50, price_refinement_window=5):
    """
    Sequential P0 selection with dynamic extrema correction and dual filtering
    """
    print(f"=== Sequential RSI P0 + Dynamic Extrema Correction ===")
    print(f"Parameters: lookback_days={lookback_days}, trend_threshold={trend_threshold}, price_refinement_window={price_refinement_window}")
    
    # Calculate RSI once for the full dataset (FIX: was calculating on search windows)
    full_rsi = calculate_rsi(close_prices, period=rsi_period).dropna()
    
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
        
        # Find highest RSI point in window as P0 (using full dataset RSI)
        search_window_rsi = full_rsi.loc[search_data.index[0]:search_data.index[-1]]
        if search_window_rsi.isna().all():
            print("No valid RSI data. Advancing position.")
            current_position += lookback_days // 2
            continue
            
        rsi_p0_date = search_window_rsi.idxmax()
        p0_rsi = search_window_rsi.max()
        
        # Apply bidirectional price refinement to P0
        p0_date = refine_peak_with_price(close_prices, rsi_p0_date, price_refinement_window)
        p0_price = close_prices.loc[p0_date]
        
        print(f"P0 selected: RSI peak at {rsi_p0_date.date()} (RSI: {p0_rsi:.2f})")
        if p0_date != rsi_p0_date:
            print(f"P0 refined to: {p0_date.date()} (${p0_price:.2f})")
        
        # Dynamic wave building with extrema correction
        wave = build_wave_with_dynamic_correction(
            close_prices, p0_date, full_rsi, rsi_drop_threshold, rsi_rise_ratio, trend_threshold, price_refinement_window
        )
        
        if wave:
            all_waves.append(wave)
            
            # Get trigger points for this wave period
            wave_start_date = close_prices.index[wave['indices'][0]]
            wave_end_date = close_prices.index[wave['indices'][-1]]
            
            # Find extremas in this period to get trigger points
            wave_data = close_prices.loc[wave_start_date:wave_end_date]
            wave_rsi = full_rsi.loc[wave_start_date:wave_end_date]
            _, wave_triggers = find_extremas_with_rsi(
                wave_data, wave_rsi, rsi_period, rsi_drop_threshold, rsi_rise_ratio, price_refinement_window
            )
            all_trigger_points.extend(wave_triggers)
            
            # Advance past this wave
            wave_end_idx = wave['indices'][-1]
            current_position = wave_end_idx + 1
            print(f"Wave found! Advancing to position {current_position}")
        else:
            print("No valid wave found with dynamic correction. Advancing position.")
            current_position += lookback_days // 2
    
    print(f"\n=== Sequential Detection Complete ===")
    print(f"Total valid waves: {len(all_waves)}")
    
    return all_waves, all_trigger_points

# =========================================================
# Plotting functions (simplified from latest_code.py)
# =========================================================

def plot_overview_chart(close_prices, all_waves, rsi_series, all_trigger_points, rsi_drop_threshold, rsi_rise_ratio, 
                        lookback_days=50, rsi_period=14, trend_threshold=0.95, price_refinement_window=5):
    """
    Plot overview chart showing all detected waves with RSI subplot
    """
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    
    start_date_all = close_prices.index.min().date()
    end_date_all = close_prices.index.max().date()
    
    # Subplot 1: Price Chart
    ax1.plot(close_prices.index.values, close_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
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

        ax1.plot(wave_points_dates.values, close_prices.iloc[wave_indices].values, f'{color[0]}o-', markersize=6, label=label)
        ax1.annotate(f'{wave_type.capitalize()} Wave {i+1}', 
                    (wave_points_dates[0], close_prices.iloc[wave_indices[0]]), 
                    xytext=(5, 10), textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold')
        
    ax1.legend(loc='upper right')
    title_str = f'BTC Price Chart: {start_date_all} to {end_date_all} (Waves Marked)\n'
    title_str += f'RSI Drop Threshold: {rsi_drop_threshold}, RSI Rise Ratio: {rsi_rise_ratio:.2f}'
    ax1.set_title(title_str, fontsize=16)
    ax1.set_ylabel('Close Price', fontsize=12)
    ax1.grid(True)
    
    # Add parameter textbox
    param_text = f"""Parameters:
• Lookback Days: {lookback_days}
• RSI Period: {rsi_period}
• RSI Drop Threshold: {rsi_drop_threshold}
• RSI Rise Ratio: {rsi_rise_ratio:.3f}
• Trend Threshold: {trend_threshold}
• Price Refinement Window: {price_refinement_window}"""
    
    ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Subplot 2: RSI Chart
    ax2.plot(rsi_series.index.values, rsi_series.values, color='purple', label='14-Day RSI', linewidth=1.5)
    ax2.axhline(70, linestyle='--', color='lightcoral', alpha=0.7)
    ax2.axhline(30, linestyle='--', color='lightgreen', alpha=0.7)
    
    # Plot trigger points (avoid duplicates)
    plotted_triggers = set()
    
    def get_first_last_triggers(wave_indices, all_trigger_points, close_prices):
        wave_start_date = close_prices.index[wave_indices[0]]
        wave_end_date = close_prices.index[wave_indices[-1]]
        
        wave_triggers = [p for p in all_trigger_points if p['date'] >= wave_start_date and p['date'] <= wave_end_date]
        
        first_trigger = wave_triggers[0] if wave_triggers else None
        last_trigger = wave_triggers[-1] if wave_triggers else None
        
        return first_trigger, last_trigger
    
    # Plot trigger points for all waves
    for wave in all_waves:
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
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_individual_wave(file_path, close_prices, rsi_series, wave_indices, trigger_points, plot_range_days=30, wave_number=1, wave_type="", if_plot_rsi=True):
    """
    Plot individual wave with optional RSI subplot
    """
    wave_start_date = close_prices.index[wave_indices[0]] 
    wave_end_date = close_prices.index[wave_indices[-1]]
    
    start_date = wave_start_date - pd.Timedelta(days=plot_range_days)
    end_date = wave_end_date + pd.Timedelta(days=plot_range_days)
    
    plot_prices = close_prices.loc[start_date:end_date]
    
    # Conditionally create subplots
    if if_plot_rsi:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        ax1.plot(plot_prices.index.values, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(15, 8))
        ax2 = None
        ax1.plot(plot_prices.index.values, plot_prices.values, label='Close Price', color='blue', linewidth=1.5, alpha=0.7)
    
    wave_points_dates = close_prices.index[wave_indices]
    wave_points_prices = close_prices.iloc[wave_indices].values
    color = 'red' if wave_type == 'strict' else 'green' if wave_type == 'relaxed' else 'magenta'
    ax1.plot(wave_points_dates.values, wave_points_prices, 'o-', color=color, label=f'{wave_type.capitalize()} Wave', markersize=6)
    
    # Point annotations
    for j in range(len(wave_indices)):
        ax1.annotate(f'P{j+1}', (wave_points_dates[j], wave_points_prices[j]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=10, color=color, fontweight='bold')
    
    # Plot trigger points on price chart
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
        ax2.plot(plot_rsi.index.values, plot_rsi.values, color='purple', label='14-Day RSI', linewidth=1.5)
        wave_points_rsi = rsi_series.loc[wave_points_dates].values
        ax2.plot(wave_points_dates.values, wave_points_rsi, 'o', color=color, markersize=6)
        
        for j in range(len(wave_indices)):
            ax2.annotate(f'P{j+1} ({wave_points_rsi[j]:.2f})', 
                        (wave_points_dates[j], wave_points_rsi[j]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=8, color=color, fontweight='bold')
        
        # Plot trigger points on RSI chart
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
# Main function
# =========================================================

def main(file_path='BTC.csv', lookback_days=50, rsi_period=14, rsi_drop_threshold=10, 
         rsi_rise_ratio=1/3, trend_threshold=0.95, recent_days=50, price_refinement_window=5):
    print("=== BTC Wave Detection System (Approach 2) ===")
    print("Sequential P0 + Dynamic Extrema Correction with Dual Filtering")
    
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
    all_waves, all_trigger_points = sequential_wave_detection(
        close_prices, lookback_days, rsi_period, rsi_drop_threshold, 
        rsi_rise_ratio, trend_threshold, recent_days, price_refinement_window
    )
    
    # Results summary
    print(f"\n=== Results ===")
    print(f"Total waves detected: {len(all_waves)}")
    for i, wave in enumerate(all_waves):
        trend_info = wave['trend_info']
        print(f"Wave {i+1}: {wave['type']} - {trend_info['start_date']} to {trend_info['end_date']}")
    
    # Plot overview chart
    if all_waves:
        print("\nGenerating overview chart...")
        plot_overview_chart(close_prices, all_waves, rsi_series, all_trigger_points, rsi_drop_threshold, rsi_rise_ratio,
                           lookback_days, rsi_period, trend_threshold, price_refinement_window)
        
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
            
            plot_individual_wave('BTC.csv', close_prices, rsi_series, wave_indices, wave_trigger_points, 
                               plot_range_days=30, wave_number=i+1, wave_type=wave_type, if_plot_rsi=True)

if __name__ == "__main__":
    main()