// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/phillox_generator.hpp"

namespace ov {
namespace reference {
namespace phillox {

// ====== Utils functions ======
namespace {

// Splits uint64 value into two uint32 values with right and left part of original value.
std::pair<uint32_t, uint32_t> split_high_low(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

// Splits uint64 global_seed value into two uint32 values to create a key.
std::array<uint32_t, 2> split_into_key(uint64_t global_seed) {
    std::array<uint32_t, 2> key;
    key[0] = static_cast<uint32_t>(global_seed);
    key[1] = static_cast<uint32_t>(global_seed >> 32);
    return key;
}

// Splits two uint64 values (adjusted for previous state) into four uint32 values to create a counter.
std::array<uint32_t, 4> split_into_counter(std::pair<uint64_t, uint64_t> previous_state, uint64_t operator_seed) {
    std::array<uint32_t, 4> counter;
    counter[0] = static_cast<uint32_t>(previous_state.first);
    counter[1] = static_cast<uint32_t>(previous_state.first >> 32);
    if (previous_state.second == 0) {
        counter[2] = static_cast<uint32_t>(operator_seed);
        counter[3] = static_cast<uint32_t>(operator_seed >> 32);
    } else {
        counter[2] = static_cast<uint32_t>(previous_state.second);
        counter[3] = static_cast<uint32_t>(previous_state.second >> 32);
    }
    return counter;
}

// Helper function to return the lower and higher 32-bits from two 32-bit integer multiplications.
void multiply_high_low(uint32_t a, uint32_t b, uint32_t* result_low, uint32_t* result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> 32);
}

uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & 0x80000000) | (v & 0x7fffffff);
}

uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? 0x9908b0df : 0);
}

}  // namespace

// ====== PhilloxGenerator base class functions ======

PhilloxGenerator::PhilloxGenerator(const PhilloxAlignment alignment,
                                   const uint64_t global_seed,
                                   const uint64_t operator_seed,
                                   const std::pair<uint64_t, uint64_t> previous_state,
                                   const size_t generated_elements_count)
    : m_alignment(alignment),
      m_global_seed(global_seed),
      m_operator_seed(operator_seed),
      m_previous_state(previous_state),
      m_generated_elements_count(generated_elements_count) {}

uint64_t PhilloxGenerator::get_global_seed() const {
    return m_global_seed;
}

uint64_t PhilloxGenerator::get_operator_seed() const {
    return m_operator_seed;
}

std::pair<uint64_t, uint64_t> PhilloxGenerator::get_previous_state() const {
    return m_previous_state;
}

size_t PhilloxGenerator::get_generated_elements_count() const {
    return m_generated_elements_count;
}

PhilloxAlignment PhilloxGenerator::get_alignment() const {
    return m_alignment;
}

void PhilloxGenerator::set_global_seed(const uint64_t global_seed) {
    m_global_seed = global_seed;
}

void PhilloxGenerator::set_operator_seed(const uint64_t op_seed) {
    m_operator_seed = op_seed;
}

// ====== MockPhilloxGenerator functions ======
MockPhilloxGenerator::MockPhilloxGenerator() : PhilloxGenerator(PhilloxAlignment::MOCK, 0, 0, {0, 0}, 1){};

std::pair<uint64_t, uint64_t> MockPhilloxGenerator::get_next_state() {
    return get_previous_state();
}

PhilloxOutput MockPhilloxGenerator::random() {
    PhilloxOutput result(get_generated_elements_count(), 0);
    return result;
}

// ====== OpenvinoPhilloxGenerator functions ======
OpenvinoPhilloxGenerator::OpenvinoPhilloxGenerator(const uint64_t global_seed,
                                                   const uint64_t operator_seed,
                                                   const std::pair<uint64_t, uint64_t> previous_state)
    : PhilloxGenerator(PhilloxAlignment::OPENVINO,
                       global_seed,
                       previous_state.second > 0 ? previous_state.second : operator_seed,
                       previous_state,
                       4UL),
      m_n64(previous_state.first),
      m_key64(global_seed),
      m_counter64(previous_state.second > 0 ? previous_state.second : operator_seed),
      m_n(split_high_low(m_n64)),
      m_key(split_high_low(m_key64)),
      m_counter(split_high_low(m_counter64)),
      m_total_generated_elements(0) {}

void OpenvinoPhilloxGenerator::increment_key() {
    m_key.first += CRUSH_RESISTANCE_LOWER_VALUE;
    m_key.second += CRUSH_RESISTANCE_UPPER_VALUE;
}

void OpenvinoPhilloxGenerator::execute_single_round() {
    // Each round performs following updating for n and counter:
    // left uint32 part = mullo(R, M)
    // right uint32 part  = mulhi(R, M) xor k xor L
    // mulhi(a, b) = floor((a * b) / 2^32)
    // mullo(a, b) = (a * b) mod 2^32,
    // where M - statistic_maximizing_multiplier const
    auto prod0 = split_high_low(STATISTIC_MAXIMIZING_MULTIPLIER_N * m_n.first);
    auto prod1 = split_high_low(STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER * m_counter.first);
    m_n.first = prod1.second ^ m_n.second ^ m_key.first;
    m_n.second = prod1.first;
    m_counter.first = prod0.second ^ m_counter.second ^ m_key.second;
    m_counter.second = prod0.first;
}

std::pair<uint64_t, uint64_t> OpenvinoPhilloxGenerator::get_next_state() {
    // 256 is chosen to match Tensorflow's 'Reserve128Samples()' behavior
    uint64_t skip_count = m_total_generated_elements * 256;
    uint64_t new_n = get_previous_state().first + skip_count;
    uint64_t new_op_seed = get_previous_state().second + (new_n < skip_count ? 1 : 0);

    return {new_n, new_op_seed};
}

PhilloxOutput OpenvinoPhilloxGenerator::random() {
    PhilloxOutput result(get_generated_elements_count());

    for (uint16_t i = 0; i < 9; ++i) {
        execute_single_round();
        increment_key();
    }
    execute_single_round();

    result[0] = m_n.first;
    result[1] = m_n.second;
    result[2] = m_counter.first;
    result[3] = m_counter.second;

    if (++m_n64 == 0) {
        ++m_counter64;
    }

    m_n = split_high_low(m_n64);
    m_counter = split_high_low(m_counter64);

    ++m_total_generated_elements;

    return result;
}

// ====== PytorchPhilloxGenerator functions ======
PytorchPhilloxGenerator::PytorchPhilloxGenerator(const uint64_t global_seed)
    : PhilloxGenerator(PhilloxAlignment::PYTORCH, global_seed, 0UL, {0UL, 0UL}, 2UL),
      m_left(1),
      m_next(0) {
    m_mersenne_state[0] = global_seed & 0xffffffff;
    for (uint32_t j = 1; j < MERSENNE_STATE_N; ++j) {
        m_mersenne_state[j] = (1812433253 * (m_mersenne_state[j - 1] ^ (m_mersenne_state[j - 1] >> 30)) + j);
    }
}

std::pair<uint64_t, uint64_t> PytorchPhilloxGenerator::get_next_state() {
    return get_previous_state();
}

void PytorchPhilloxGenerator::next_mersenne_state() {
    uint32_t* p = m_mersenne_state.data();
    m_left = MERSENNE_STATE_N;
    m_next = 0;

    for (int j = MERSENNE_STATE_N - MERSENNE_STATE_M + 1; --j; p++) {
        *p = p[MERSENNE_STATE_M] ^ twist(p[0], p[1]);
    }

    for (int j = MERSENNE_STATE_M; --j; p++) {
        *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], p[1]);
    }

    *p = p[MERSENNE_STATE_M - MERSENNE_STATE_N] ^ twist(p[0], m_mersenne_state[0]);
}

PhilloxOutput PytorchPhilloxGenerator::random() {
    PhilloxOutput result(get_generated_elements_count());

    if (--m_left == 0) {
        next_mersenne_state();
    }

    result[0] = *(m_mersenne_state.data() + m_next++);
    result[0] ^= (result[0] >> 11);
    result[0] ^= (result[0] << 7) & 0x9d2c5680;
    result[0] ^= (result[0] << 15) & 0xefc60000;
    result[0] ^= (result[0] >> 18);

    if (--m_left == 0) {
        next_mersenne_state();
    }

    result[1] = *(m_mersenne_state.data() + m_next++);
    result[1] ^= (result[1] >> 11);
    result[1] ^= (result[1] << 7) & 0x9d2c5680;
    result[1] ^= (result[1] << 15) & 0xefc60000;
    result[1] ^= (result[1] >> 18);

    if (--m_left == 0) {
        next_mersenne_state();
    }

    return result;
}

// ====== TensorflowPhilloxGenerator functions ======

TensorflowPhilloxGenerator::TensorflowPhilloxGenerator(const uint64_t global_seed,
                                                       const uint64_t operator_seed,
                                                       const std::pair<uint64_t, uint64_t> previous_state)
    : PhilloxGenerator(PhilloxAlignment::TENSORFLOW, global_seed, operator_seed, previous_state, 4UL),
      m_key(split_into_key(global_seed)),
      m_counter(split_into_counter(previous_state, operator_seed)) {}

std::pair<uint64_t, uint64_t> TensorflowPhilloxGenerator::get_next_state() {
    return {(static_cast<uint64_t>(m_counter[1]) << 32) + m_counter[0],
            (static_cast<uint64_t>(m_counter[3]) << 32) + m_counter[2]};
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

void TensorflowPhilloxGenerator::raise_key(std::array<uint32_t, 2>& key) {
    key[0] += CRUSH_RESISTANCE_LOWER_VALUE;
    key[1] += CRUSH_RESISTANCE_UPPER_VALUE;
}

void TensorflowPhilloxGenerator::compute_single_round(std::array<uint32_t, 2>& key, std::array<uint32_t, 4>& counter) {
    uint32_t lo0;
    uint32_t hi0;
    multiply_high_low(STATISTIC_MAXIMIZING_MULTIPLIER_N, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    multiply_high_low(STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER, counter[2], &lo1, &hi1);

    counter[0] = hi1 ^ counter[1] ^ key[0];
    counter[1] = lo1;
    counter[2] = hi0 ^ counter[3] ^ key[1];
    counter[3] = lo0;
}

PhilloxOutput TensorflowPhilloxGenerator::random() {
    // Logic very similar to OpenVINO's, the main difference is behavior of key and counter
    // Here, key and counter is copied before execution, and skip_one increments the original values,
    // instead of incrementing the 'twisted' ones.
    PhilloxOutput result(get_generated_elements_count());

    auto key = m_key;
    auto counter = m_counter;

    for (uint16_t i = 0; i < 9; ++i) {
        compute_single_round(key, counter);
        raise_key(key);
    }
    compute_single_round(key, counter);

    result[0] = counter[0];
    result[1] = counter[1];
    result[2] = counter[2];
    result[3] = counter[3];

    skip_one();

    return result;
}

// ====== General selector function to construct a desired generator  ======

std::shared_ptr<PhilloxGenerator> make_phillox_generator(uint64_t seed,
                                                         uint64_t seed2,
                                                         std::pair<uint64_t, uint64_t> prev_state,
                                                         size_t elem_count,
                                                         PhilloxAlignment alignment) {
    switch (alignment) {
    case PhilloxAlignment::OPENVINO:
        return std::make_shared<OpenvinoPhilloxGenerator>(seed, seed2, prev_state);
    case PhilloxAlignment::TENSORFLOW:
        return std::make_shared<TensorflowPhilloxGenerator>(seed, seed2, prev_state);
    case PhilloxAlignment::PYTORCH:
        return std::make_shared<PytorchPhilloxGenerator>(seed);
    default:
        return std::make_shared<MockPhilloxGenerator>();
    }
}

}  // namespace phillox
}  // namespace reference
}  // namespace ov
