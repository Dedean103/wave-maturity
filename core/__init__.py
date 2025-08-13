"""
核心功能模块包
包含波浪检测和 RSI 分析的核心逻辑
"""

from .wave_detection import (
    find_all_wave_patterns,
    remove_overlapping_waves,
    is_downtrend_five_wave_strict,
    is_downtrend_five_wave_relaxed
)

from .rsi_analysis import find_extremas_with_rsi

__all__ = [
    'find_all_wave_patterns',
    'remove_overlapping_waves',
    'is_downtrend_five_wave_strict',
    'is_downtrend_five_wave_relaxed',
    'find_extremas_with_rsi'
]
