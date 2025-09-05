#!/usr/bin/env python3
"""
Direct RSI analysis around target dates
"""

import pandas as pd
from current_code import calculate_rsi

def check_rsi_around_targets():
    # Load data
    df = pd.read_csv('BTC.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    close_prices = df['close']
    
    # Calculate RSI
    full_rsi = calculate_rsi(close_prices, period=14).dropna()
    
    target_dates = ['2024-03-11', '2024-07-28']
    
    for target_str in target_dates:
        target_date = pd.to_datetime(target_str)
        print(f"\n=== Target Date: {target_str} ===")
        
        # Find 30-day window around target
        start_window = target_date - pd.Timedelta(days=30)
        end_window = target_date + pd.Timedelta(days=30)
        
        try:
            window_data = full_rsi.loc[start_window:end_window]
            if len(window_data) > 0:
                target_rsi = full_rsi.loc[target_date] if target_date in full_rsi.index else None
                max_rsi = window_data.max()
                max_rsi_date = window_data.idxmax()
                
                print(f"Target RSI: {target_rsi:.2f} on {target_date.date()}")
                print(f"Window max RSI: {max_rsi:.2f} on {max_rsi_date.date()}")
                print(f"Target price: ${close_prices.loc[target_date]:.0f}")
                
                # Show top 5 RSI values in window
                top_5 = window_data.nlargest(5)
                print("Top 5 RSI in window:")
                for date, rsi in top_5.items():
                    marker = "← TARGET" if date.date() == target_date.date() else ""
                    print(f"  {date.date()}: RSI {rsi:.2f} (${close_prices.loc[date]:.0f}) {marker}")
                    
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    check_rsi_around_targets()