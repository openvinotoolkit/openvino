// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/phillox_generator.hpp"

#include <stdio.h>

namespace ov {

namespace reference {

namespace phillox {

namespace {

// ====== OPENVINO HELPERS ======

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

// Splits uint64 value into two uint32 values with right and left part of original value.
std::pair<uint32_t, uint32_t> split_high_low(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

// ====== PYTORCH HELPERS ======

uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & 0x80000000) | (v & 0x7fffffff);
}

inline uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? 0x9908b0df : 0);
}

// ====== TENSORFLOW HELPERS ======

// Helper function to return the lower and higher 32-bits from two 32-bit integer multiplications.
void multiply_high_low(uint32_t a, uint32_t b, uint32_t* result_low, uint32_t* result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> 32);
}

}  // namespace

// ====== OPENVINO MAIN FUNCTIONS ======
void OpenvinoPhilloxGenerator::execute_single_round() {
    // Split key, counter and n into two uint32 values.
    auto n_lr = split_high_low(m_n);
    auto counter_lr = split_high_low(m_operator_seed);
    auto key_lr = split_high_low(m_global_seed);

    // Each round performs following updating for n and counter:
    // left uint32 part = mullo(R, M)
    // right uint32 part  = mulhi(R, M) xor k xor L
    // mulhi(a, b) = floor((a * b) / 2^32)
    // mullo(a, b) = (a * b) mod 2^32,
    // where M - statistic_maximizing_multiplier const
    auto prod0 = split_high_low(m_statistic_maximizing_multiplier_n * n_lr.first);
    auto prod1 = split_high_low(m_statistic_maximizing_multiplier_counter * counter_lr.first);
    n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
    n_lr.second = prod1.first;
    counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
    counter_lr.second = prod0.first;

    // Unite counter and n into uint64 values.
    m_n = unite_high_low(n_lr.second, n_lr.first);
    m_operator_seed = unite_high_low(counter_lr.second, counter_lr.first);
}

void OpenvinoPhilloxGenerator::increment_key() {
    auto key_lr = split_high_low(m_global_seed);
    key_lr.first += m_crush_resistance_const_lower_value;
    key_lr.second += m_crush_resistance_const_upper_value;
    m_global_seed = unite_high_low(key_lr.second, key_lr.first);
}

PhilloxOutput OpenvinoPhilloxGenerator::random() {
    PhilloxOutput result(get_output_size());

    // Runs single "round" of Philox algorithm a preset amount of times (10)
    // increase key value after each iteration except the last.
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();
    increment_key();
    execute_single_round();

    auto res1 = split_high_low(m_n);
    auto res2 = split_high_low(m_operator_seed);
    result[0] = res1.first;
    result[1] = res1.second;
    result[2] = res2.first;
    result[3] = res2.second;

    auto n_low = res1.first;
    auto n_high = res1.second;
    auto counter_low = res2.first;
    auto counter_high = res2.second;

    if (++n_low == 0) {
        if (++n_high == 0) {
            if (++counter_low == 0) {
                ++counter_high;
            }
        }
    }
    m_n = unite_high_low(n_high, n_low);
    m_operator_seed = unite_high_low(counter_high, counter_low);

    ++m_generated_elements_count;

    return result;
}

// ====== PYTORCH MAIN FUNCTIONS ======
void PytorchPhilloxGenerator::next_state() {
    uint32_t* p = m_state.data();
    m_left = _PYTORCH_MERSENNE_STATE_N;
    m_next = 0;

    for (int j = _PYTORCH_MERSENNE_STATE_N - _PYTORCH_MERSENNE_STATE_M + 1; --j; p++) {
        *p = p[_PYTORCH_MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for (int j = _PYTORCH_MERSENNE_STATE_M; --j; p++) {
        *p = p[_PYTORCH_MERSENNE_STATE_M - _PYTORCH_MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[_PYTORCH_MERSENNE_STATE_M - _PYTORCH_MERSENNE_STATE_N] ^ twist(p[0], m_state[0]);
}

PhilloxOutput PytorchPhilloxGenerator::random() {
    PhilloxOutput result(get_output_size());

    if (--m_left == 0) {
        next_state();
    }

    result[0] = *(m_state.data() + m_next++);
    result[0] ^= (result[0] >> 11);
    result[0] ^= (result[0] << 7) & 0x9d2c5680;
    result[0] ^= (result[0] << 15) & 0xefc60000;
    result[0] ^= (result[0] >> 18);

    if (--m_left == 0) {
        next_state();
    }

    result[1] = *(m_state.data() + m_next++);
    result[1] ^= (result[1] >> 11);
    result[1] ^= (result[1] << 7) & 0x9d2c5680;
    result[1] ^= (result[1] << 15) & 0xefc60000;
    result[1] ^= (result[1] >> 18);

    if (--m_left == 0) {
        next_state();
    }

    return result;
}

// ====== TENSORFLOW MAIN FUNCTIONS ======
void TensorflowPhilloxGenerator::skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    m_counter[0] += count_lo;
    if (m_counter[0] < count_lo) {
        ++count_hi;
    }

    m_counter[1] += count_hi;
    if (m_counter[1] < count_hi) {
        if (++m_counter[2] == 0) {
            ++m_counter[3];
        }
    }
}

void TensorflowPhilloxGenerator::skip_one() {
    if (++m_counter[0] == 0) {
        if (++m_counter[1] == 0) {
            if (++m_counter[2] == 0) {
                ++m_counter[3];
            }
        }
    }
}

void TensorflowPhilloxGenerator::skip_256() {
    for (uint16_t i = 0; i < 256; ++i) {
        skip_one();
    }
}

void TensorflowPhilloxGenerator::raise_key(std::array<uint32_t, 2>& key) {
    key[0] += _TENSORFLOW_kPhiloxW32A;
    key[1] += _TENSORFLOW_kPhiloxW32B;
}

void TensorflowPhilloxGenerator::compute_single_round(std::array<uint32_t, 2>& key, std::array<uint32_t, 4>& counter) {
    uint32_t lo0;
    uint32_t hi0;
    multiply_high_low(_TENSORFLOW_kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    multiply_high_low(_TENSORFLOW_kPhiloxM4x32B, counter[2], &lo1, &hi1);

    counter[0] = hi1 ^ counter[1] ^ key[0];
    counter[1] = lo1;
    counter[2] = hi0 ^ counter[3] ^ key[1];
    counter[3] = lo0;
}

PhilloxOutput TensorflowPhilloxGenerator::random() {
    PhilloxOutput result(get_output_size());

    // Run the single rounds for ten times. Manually unrolling the loop
    // for better performance.
    auto key = m_key;
    auto counter = m_counter;

    std::cout << "S: " << key[0] << " " << key[1] << " " << counter[0] << " " << counter[1] << " " << counter[2] << " "
              << counter[3] << " " << std::endl;

    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);
    raise_key(key);
    compute_single_round(key, counter);

    result[0] = counter[0];
    result[1] = counter[1];
    result[2] = counter[2];
    result[3] = counter[3];

    std::cout << "B: " << key[0] << " " << key[1] << " " << counter[0] << " " << counter[1] << " " << counter[2] << " "
              << counter[3] << " " << std::endl;
    skip_one();
    // skip_256();
    std::cout << "A: " << m_key[0] << " " << m_key[1] << " " << m_counter[0] << " " << m_counter[1] << " "
              << m_counter[2] << " " << m_counter[3] << " " << std::endl;

    return result;
}

}  // namespace phillox
}  // namespace reference
}  // namespace ov
