#!/usr/bin/env python3
"""
Enhanced current_code.py output with RSI values for detected waves
"""

import pandas as pd
from current_code import sequential_wave_detection, calculate_rsi

def enhanced_current_code_analysis():
    print("=== ENHANCED CURRENT_CODE.PY ANALYSIS WITH RSI VALUES ===\n")
    
    # Load data
    df = pd.read_csv('BTC.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    close_prices = df['close']
    
    # Calculate RSI
    rsi_series = calculate_rsi(close_prices, period=14).dropna()
    
    # Run detection
    waves = sequential_wave_detection(close_prices)
    
    print(f"CURRENT_CODE.PY RESULTS WITH RSI VALUES:")
    print(f"Total waves detected: {len(waves)}")
    print()
    
    for i, wave in enumerate(waves):
        if isinstance(wave, dict) and 'indices' in wave:
            indices = wave['indices']
        else:
            # Handle tuple format (legacy)
            indices = wave
            
        wave_dates = [close_prices.index[idx] for idx in indices]
        wave_prices = [close_prices.iloc[idx] for idx in indices]
        wave_rsi_values = []
        
        for idx in indices:
            if idx < len(rsi_series):
                rsi_val = rsi_series.iloc[idx]
                wave_rsi_values.append(f"RSI:{rsi_val:.1f}")
            else:
                wave_rsi_values.append("RSI:N/A")
        
        print(f"Wave {i+1} ({'strict/relaxed' if isinstance(wave, dict) else 'legacy'}):")
        print(f"  Dates: {' -> '.join([d.strftime('%Y-%m-%d') for d in wave_dates])}")
        print(f"  Prices: {' -> '.join([f'${p:.0f}' for p in wave_prices])}")
        print(f"  RSI: {' -> '.join(wave_rsi_values)}")
        print()
    
    # Check for target waves
    print("=== TARGET WAVE ANALYSIS ===")
    target_ranges = [
        ("2024-03-11", "2024-04-17"),
        ("2024-07-28", "2024-09-06")
    ]
    
    for target_start, target_end in target_ranges:
        found = False
        for i, wave in enumerate(waves):
            if isinstance(wave, dict) and 'indices' in wave:
                indices = wave['indices']
            else:
                indices = wave
                
            wave_start = close_prices.index[indices[0]].strftime('%Y-%m-%d')
            wave_end = close_prices.index[indices[-1]].strftime('%Y-%m-%d')
            
            if (target_start <= wave_start <= target_end) or (target_start <= wave_end <= target_end):
                print(f"✓ Found target wave: {wave_start} to {wave_end}")
                found = True
                break
        
        if not found:
            print(f"✗ Target wave {target_start} to {target_end} NOT FOUND")

if __name__ == "__main__":
    enhanced_current_code_analysis()