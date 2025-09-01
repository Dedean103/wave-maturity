# Wave Analysis Logic Requirements

## Overview
Update the wave analysis logic to use RSI-based starting point detection with the following specifications:

## Core Requirements

### 1. Starting Point (P0) Selection Logic
- **RSI Calculation**: Calculate 14-day RSI for the dataset
- **Lookback Period**: Use the most recent `x` days (default: 50 days) of **remaining unused data** after each completed wave
- **P0 Selection**: Identify the highest RSI point within this lookback period as the initial starting point
- **Price Refinement**: Apply spot adjustment using `price_refinement_window` parameter
  - If a higher price exists within the refinement window, update to that point as the final P0

### 2. Wave Constraints
- **Sequential Requirement**: New P0 must be chronologically after all previous waves
- **Minimum Gap**: Maintain the same minimum gap requirement as the existing `recent_days` parameter
- **Overlap Handling**: Apply the same overlap handling logic as current implementation

### 3. Subsequent Points Logic
- **Unchanged**: The logic for finding subsequent points in each wave (P1, P2, etc.) remains the same as current implementation
- **Only P0 detection changes**: All other wave detection parameters and methods stay identical

## Parameters
- `x` (lookback_days): Number of recent days to analyze for P0 detection (default: 50)
- `price_refinement_window`: Window size for price-based P0 adjustment (existing parameter)
- `recent_days`: Minimum gap requirement between waves (existing parameter)

## Implementation Notes
- The `rsi_series` should be the 14-day RSI values
- After each completed wave, use the next 50 days of unused data for the subsequent P0 search
- Ensure no data overlap between wave searches
- Maintain all existing validation and error handling

### 4. Benchmark Validation
- **Reference File**: Use `benchmark.txt` as the reference for expected wave count
- **Quality Threshold**: If the output wave count is less than 60% of the benchmark count, flag for logic improvement
- **Action Required**: When below threshold, the algorithm needs refinement or parameter adjustment

## Quality Control
- Compare final wave count against benchmark.txt
- If `detected_waves < (benchmark_count * 0.8)`, indicate that logic improvement is needed
- Provide feedback on detection rate vs. benchmark expectations
