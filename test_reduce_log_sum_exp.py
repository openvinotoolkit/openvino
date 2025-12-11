#!/usr/bin/env python3
"""
Test script to demonstrate the numerical stability of ReduceLogSumExp implementation
This shows the difference between naive and stable implementations.
"""

import numpy as np
import math

def naive_log_sum_exp(x):
    """Naive implementation: log(sum(exp(x))) - prone to overflow"""
    return np.log(np.sum(np.exp(x)))

def stable_log_sum_exp(x):
    """Numerically stable implementation: k + log(sum(exp(x - k))) where k = max(x)"""
    k = np.max(x)
    return k + np.log(np.sum(np.exp(x - k)))

def test_numerical_stability():
    """Test numerical stability with various input ranges"""
    print("ReduceLogSumExp Numerical Stability Test")
    print("=" * 50)
    print()
    
    test_cases = [
        ("Small values", [1.0, 2.0, 3.0]),
        ("Medium values", [10.0, 11.0, 12.0]),
        ("Large values (problematic for naive)", [87.0, 88.0, 89.0]),
        ("Very large values (overflow for naive)", [100.0, 101.0, 102.0]),
        ("Mixed values", [-10.0, 0.0, 50.0, 100.0]),
    ]
    
    for name, values in test_cases:
        print(f"{name}: {values}")
        
        x = np.array(values, dtype=np.float32)
        
        # Try naive implementation
        try:
            naive_result = naive_log_sum_exp(x)
            naive_finite = np.isfinite(naive_result)
        except (OverflowError, RuntimeWarning):
            naive_result = float('inf')
            naive_finite = False
        
        # Stable implementation
        stable_result = stable_log_sum_exp(x)
        stable_finite = np.isfinite(stable_result)
        
        print(f"  Naive implementation:  {naive_result:.6f} {'✓' if naive_finite else '✗ (overflow/inf)'}")
        print(f"  Stable implementation: {stable_result:.6f} {'✓' if stable_finite else '✗ (should not happen)'}")
        
        # Show the mathematical breakdown for stable implementation
        k = np.max(x)
        shifted = x - k
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted)
        log_sum = np.log(sum_exp)
        
        print(f"  Breakdown: max={k:.1f}, exp(x-max)={exp_shifted}, sum={sum_exp:.6f}, final={k:.1f}+{log_sum:.6f}")
        print()

def demonstrate_overflow_threshold():
    """Show where naive implementation starts to fail"""
    print("Finding overflow threshold for naive implementation")
    print("=" * 50)
    
    # Test increasing values to find where naive implementation fails
    for val in [80, 85, 87, 88, 89, 90, 95, 100]:
        x = np.array([val, val], dtype=np.float32)
        
        # Expected result for two identical values: val + log(2)
        expected = val + np.log(2.0)
        
        # Naive
        try:
            naive_result = naive_log_sum_exp(x)
            naive_ok = np.isfinite(naive_result)
            naive_error = abs(naive_result - expected) if naive_ok else float('inf')
        except:
            naive_result = float('inf')
            naive_ok = False
            naive_error = float('inf')
        
        # Stable
        stable_result = stable_log_sum_exp(x)
        stable_ok = np.isfinite(stable_result)
        stable_error = abs(stable_result - expected) if stable_ok else float('inf')
        
        print(f"Input: [{val}, {val}]")
        print(f"  Expected: {expected:.6f}")
        print(f"  Naive:    {naive_result:.6f} {'✓' if naive_ok else '✗'} (error: {naive_error:.2e})")
        print(f"  Stable:   {stable_result:.6f} {'✓' if stable_ok else '✗'} (error: {stable_error:.2e})")
        print()

def explain_algorithm():
    """Explain why the stable algorithm works"""
    print("Why the stable algorithm works:")
    print("=" * 30)
    print()
    print("Naive:  log(sum(exp(x))) can overflow when exp(x) becomes too large")
    print("Stable: k + log(sum(exp(x - k))) where k = max(x)")
    print()
    print("Key insight: exp(x - k) ≤ exp(max(x) - max(x)) = exp(0) = 1")
    print("So exp(x - k) is always ≤ 1, preventing overflow in the exp() operation.")
    print()
    print("Mathematical equivalence:")
    print("  log(sum(exp(x))) = log(exp(k) * sum(exp(x - k)))")
    print("                   = log(exp(k)) + log(sum(exp(x - k)))")
    print("                   = k + log(sum(exp(x - k)))")
    print()

if __name__ == "__main__":
    explain_algorithm()
    test_numerical_stability()
    demonstrate_overflow_threshold()