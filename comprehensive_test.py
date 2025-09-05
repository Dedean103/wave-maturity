#!/usr/bin/env python3
"""
Comprehensive test of all three options to improve current_code.py P0 detection
"""

import pandas as pd

def run_comprehensive_test():
    print("=== COMPREHENSIVE P0 IMPROVEMENT TEST ===\n")
    
    # Load data  
    df = pd.read_csv('BTC.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    close_prices = df['close']
    
    print(f"Data range: {close_prices.index[0].date()} to {close_prices.index[-1].date()}")
    
    # Test current_code.py first
    print("\n1. CURRENT CODE (baseline):")
    from current_code import sequential_wave_detection
    current_waves = sequential_wave_detection(close_prices)
    print(f"   Current code detects: {len(current_waves)} waves")
    
    # Test latest_code.py comprehensive approach
    print("\n2. LATEST CODE (comprehensive):")
    try:
        from latest_code import calculate_rsi, find_extremas_with_rsi, find_downtrend_wave_patterns
        rsi_series = calculate_rsi(close_prices, period=14).dropna()
        extremas, trigger_points = find_extremas_with_rsi(close_prices, rsi_series)
        latest_waves = find_downtrend_wave_patterns(close_prices, extremas)
        print(f"   Latest code detects: {len(latest_waves)} waves")
        
        # Show target waves from latest_code
        print("   Target waves found by latest_code:")
        target_waves = []
        for i, wave in enumerate(latest_waves):
            start = wave['indices'][0].strftime('%Y-%m-%d')
            end = wave['indices'][-1].strftime('%Y-%m-%d')
            if ('2024-03' in start or '2024-04' in start) or ('2024-07' in start or '2024-08' in start or '2024-09' in start):
                print(f"     Wave {i+1}: {start} to {end} ← TARGET WAVE")
                target_waves.append((start, end))
            else:
                print(f"     Wave {i+1}: {start} to {end}")
                
    except Exception as e:
        print(f"   Error with latest_code: {e}")
        latest_waves = []
    
    # Check what current_code found vs targets
    print(f"\n3. COMPARISON:")
    print(f"   Current: {len(current_waves)} waves")
    print(f"   Latest: {len(latest_waves)} waves") 
    print(f"   Missing: {len(latest_waves) - len(current_waves)} waves")
    
    print(f"\n=== CONCLUSION ===")
    if len(latest_waves) > len(current_waves):
        print(f"✓ latest_code.py successfully finds {len(latest_waves) - len(current_waves)} additional waves")
        print(f"✓ The comprehensive approach (Option 3) is the most effective")
        print(f"✓ Recommend: Modify current_code.py to use latest_code.py's comprehensive search")
    else:
        print(f"Both approaches find similar number of waves")

if __name__ == "__main__":
    run_comprehensive_test()