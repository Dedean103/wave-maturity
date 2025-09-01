# CLAUDE.md

This file provides guidance when working with code in this repository.

## Project Overview

This is a Bitcoin wave detection system that implements RSI-driven Elliott Wave pattern recognition. The system identifies downtrend wave structures in cryptocurrency price data using relative strength index (RSI) indicators combined with bidirectional price refinement for precise entry/exit points.

**Current Version Features:**
- **Sequential P0 Selection**: Progressive wave detection using highest RSI points in sliding windows
- **Bidirectional Price Refinement**: Refines RSI-detected peaks/valleys using actual price data within ±window
- **Dual Filtering**: Pattern validation + trend validation (trend_ratio < 0.95) to prevent uptrend false positives
- **Real-time Trading Ready**: Designed for live trading with progressive detection approach

**Version History:**
- v1.2: Pure spot price (close) to detect waves
- v1.3: RSI to detect waves  
- v1.4: Correct RSI implementation
- v1.5: Separated plots and adjustable RSI parameters
- **Current**: Sequential P0 + bidirectional price refinement with dual filtering

## Core Architecture

### Key Files
```
current_code.py        # Main implementation - sequential P0 + progressive detection
latest_code.py         # Reference implementation (do not modify until instructed)
main.py               # Legacy implementation (not actively used)
BTC.csv               # Primary Bitcoin price data
analysis_log.txt      # Latest analysis results
```

### Key Functions (current_code.py)

1. **Sequential Wave Detection** (`sequential_wave_detection()`):
   - Progressive P0 selection using highest RSI in 50-day windows
   - Corrected position advancement: `wave_end_idx + 1` (no data skipping)
   - Dual filtering: pattern + trend validation

2. **Bidirectional Price Refinement**:
   - `refine_peak_with_price()`: Finds higher prices within ±window of RSI peaks
   - `refine_valley_with_price()`: Finds lower prices within ±window of RSI valleys
   - Applied to both P0 selection and RSI extrema detection

3. **Enhanced Plotting** (`plot_overview_chart()`, `plot_individual_wave()`):
   - Dual subplot layout (price + RSI) matching latest_code.py style
   - Trigger point annotations with RSI values
   - Dynamic point labeling (P1-P6 for waves)

## Development Commands

### Running the Analysis
```bash
# Main analysis with default parameters
python3 current_code.py

# With custom parameters (all configurable)
python3 -c "
from current_code import main
main(file_path='BTC.csv', lookback_days=50, rsi_period=14, 
     rsi_drop_threshold=10, rsi_rise_ratio=1/3, trend_threshold=0.95, 
     recent_days=50, price_refinement_window=5)
"
```

### Code Testing and Git Workflow
**IMPORTANT**: After completing any code changes, always:
1. **Test the code**: Run `python3 current_code.py` to verify functionality
2. **Commit and push changes**: Run git commit and push to save progress
3. **Never modify latest_code.py** until instructed

## Key Configuration Parameters

All parameters are configurable through main() function:
- `lookback_days = 50` - P0 search window size
- `rsi_period = 14` - RSI calculation period
- `rsi_drop_threshold = 10` - RSI drop threshold for peak detection
- `rsi_rise_ratio = 1/3` - Proportional rise threshold for valley detection
- `trend_threshold = 0.95` - Trend validation threshold (< 0.95 = downtrend)
- `price_refinement_window = 5` - Bidirectional price search window (±2-3 days)

## Current System Behavior

### Sequential P0 Selection Process
1. **Start**: From beginning of data or after previous wave
2. **P0 Search**: Find highest RSI point in next 50-day window
3. **Price Refinement**: Search ±5 days for better entry price
4. **Progressive Detection**: Find subsequent extrema from P0 onwards
5. **Dual Validation**: Check pattern rules + downtrend requirement
6. **Advance**: Move to `wave_end_idx + 1` (no data skipping)

### Data Format Requirements

CSV files must contain:
- `datetime` column - timestamps (converted to datetime index)
- `close` column - closing prices for analysis

Primary data file: `BTC.csv` (2021-12-20 to 2025-06-02, 1261 records)

## Implementation Notes

- **Approach 2**: Progressive detection (find waves as extrema are discovered)
- **No overlap handling**: Simplified compared to latest_code.py complex merging
- **Real-time compatible**: Suitable for live trading applications
- **Position advancement corrected**: No longer skips 50 days between waves
- **Bidirectional refinement**: Improves trading accuracy with precise entry/exit points

## Latest Results

Current detection: **8 valid downtrend waves** covering major Bitcoin bear market periods:
- Wave detection spans 2022-2025 data
- All waves pass dual filtering (pattern + downtrend validation)
- Price refinement active throughout detection pipeline

## Important Notes

- **NEVER mention Claude in git commit messages** - use descriptive technical messages only