# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin wave detection system that implements RSI-driven Elliott Wave pattern recognition. The system identifies downtrend five-wave and seven-wave structures in cryptocurrency price data using relative strength index (RSI) indicators to detect peaks and valleys.

**Latest Version Features:**
- **Enhanced Wave Detection**: Support for both 5-wave (6-point) and 7-wave (8-point) patterns
- **Intelligent Overlap Handling**: Automatically merges overlapping 5-wave patterns into 7-wave structures
- **Improved Architecture**: Modular design with clear separation of concerns
- **Advanced Visualization**: Dynamic plotting that adapts to different wave structures

**Version History:**
- v1.2: Pure spot price (close) to detect waves
- v1.3: RSI to detect waves  
- v1.4: Correct RSI implementation
- v1.5: Separated plots and adjustable RSI parameters
- **Latest**: Enhanced overlap detection and 7-wave pattern support

## Core Architecture

### Modular Structure
```
core/                  # Core analysis modules
├── wave_detection.py  # Wave pattern recognition, validation, and overlap handling
└── rsi_analysis.py   # RSI-driven extrema identification

utils/                 # Utility functions
└── technical_indicators.py  # RSI calculation

visualization/         # Plotting and chart generation
└── plotting.py       # Individual wave and overview charts (supports 6 & 8 point waves)

main.py                    # Main application entry point with enhanced workflow
test.ipynb               # Interactive debugging notebook
wave_maturity_test.ipynb # Additional testing notebook
latest_code.py           # Reference implementation for latest features
wave_detector_functions.py # Legacy wave detection functions
```

### Key Components

1. **Enhanced Wave Detection Engine** (`core/wave_detection.py`):
   - `find_downtrend_wave_patterns()`: Generalized function supporting 6-point and 8-point waves
   - `is_downtrend_seven_wave()`: New 7-wave pattern validation
   - `handle_overlapping_waves()`: Smart overlap resolution with merging capability
   - Legacy functions maintained for backward compatibility

2. **RSI Analysis Engine** (`core/rsi_analysis.py`):
   - Uses RSI drop/rise triggers to identify price extrema
   - Dynamic threshold calculation based on wave amplitude
   - Configurable RSI period, drop threshold, and rise ratio

3. **Enhanced Visualization System** (`visualization/plotting.py`):
   - `plot_overview_chart()`: Shows all wave types (strict, relaxed, merged)
   - `plot_individual_wave()`: Adapts to both 6-point and 8-point waves
   - Improved color coding: Red (strict), Green (relaxed), Magenta (merged)

## Development Commands

### Running the Analysis
```bash
# Main analysis (uses BTC.csv by default)
python main.py

# Interactive debugging and parameter tuning
jupyter notebook test.ipynb
```

### Code Testing and Git Workflow
**IMPORTANT**: After completing any code changes, always:
1. **Test the code**: Run `python3 main.py` to verify the implementation works correctly
2. **Commit and push changes**: Automatically run git commit and push to save progress

### Testing and Development
The `test.ipynb` notebook provides:
- Step-by-step analysis execution
- Parameter sensitivity testing
- Debugging tools and statistics
- Visual inspection of each processing stage

### Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Required packages:
# - pandas>=1.3.0
# - numpy>=1.21.0  
# - matplotlib>=3.5.0
# - seaborn>=0.11.0
```

## Key Configuration Parameters

Located in `main.py` and configurable in `test.ipynb`:
- `rsi_period = 14` - RSI calculation period
- `rsi_drop_threshold = 10` - RSI drop threshold for peak detection
- `rsi_rise_ratio = 1/3` - Proportional rise threshold for valley detection

## New Features in Latest Version

### 1. Seven-Wave Pattern Detection
- Detects 8-point downtrend patterns following P1>P2<P3>P4<P5>P6<P7>P8 structure
- Additional validation: P8 < P6 (ensures proper downtrend completion)

### 2. Intelligent Overlap Handling
- Automatically detects overlapping 5-wave patterns
- Attempts to merge overlapping waves into 7-wave structures
- Falls back to removing both waves if merging fails
- Detailed logging of merge operations

### 3. Enhanced Data Structures
- Wave objects now contain `{'indices': [...], 'type': 'strict'|'relaxed'|'merged'}`
- Backward compatibility maintained with legacy tuple returns

### 4. Improved Visualization
- Dynamic point labeling (P1-P6 for 5-waves, P1-P8 for 7-waves)
- Color-coded wave types for easy identification
- Enhanced overview charts with proper legend management

## Data Format Requirements

CSV files must contain:
- `datetime` column - timestamps (will be converted to datetime index)
- `close` column - closing prices for analysis

Default data files: `BTC.csv`, `ETH.csv`, `btc_with_rsi.csv`

## Implementation Notes

- The system now prioritizes quality over quantity by merging overlapping patterns
- 7-wave patterns provide more comprehensive market structure analysis
- All legacy functions remain available for backward compatibility
- The notebook environment is optimized for parameter experimentation
- Enhanced error handling and detailed progress reporting

## Debugging and Parameter Tuning

Use `test.ipynb` for:
- Testing different RSI thresholds and ratios
- Analyzing the impact of parameter changes
- Inspecting intermediate processing steps
- Quick parameter sensitivity analysis
- Visual validation of wave detection quality
- memorize never mention claude in the git commit message