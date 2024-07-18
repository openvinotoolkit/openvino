// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/philox_generator.hpp"

namespace ov {
namespace reference {
namespace philox {

// ====== Utils functions ======
namespace {

// Splits uint64 value into two uint32 values with right and left part of original value.
std::pair<uint32_t, uint32_t> split_high_low(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

uint32_t mix_bits(uint32_t u, uint32_t v) {
    return (u & 0x80000000) | (v & 0x7fffffff);
}

uint32_t twist(uint32_t u, uint32_t v) {
    return (mix_bits(u, v) >> 1) ^ (v & 1 ? 0x9908b0df : 0);
}

}  // namespace

// ====== PhiloxGenerator base class functions ======

PhiloxGenerator::PhiloxGenerator(const op::PhiloxAlignment alignment,
                                 const uint64_t global_seed,
                                 const uint64_t operator_seed,
                                 const std::pair<uint64_t, uint64_t> previous_state)
    : m_alignment(alignment),
      m_global_seed(global_seed),
      m_operator_seed(operator_seed),
      m_previous_state(previous_state) {}

uint64_t PhiloxGenerator::get_global_seed() const {
    return m_global_seed;
}

uint64_t PhiloxGenerator::get_operator_seed() const {
    return m_operator_seed;
}

std::pair<uint64_t, uint64_t> PhiloxGenerator::get_previous_state() const {
    return m_previous_state;
}

op::PhiloxAlignment PhiloxGenerator::get_alignment() const {
    return m_alignment;
}

void PhiloxGenerator::set_global_seed(const uint64_t global_seed) {
    m_global_seed = global_seed;
}

void PhiloxGenerator::set_operator_seed(const uint64_t op_seed) {
    m_operator_seed = op_seed;
}

// ====== MockPhiloxGenerator functions ======
MockPhiloxGenerator::MockPhiloxGenerator() : PhiloxGenerator(op::PhiloxAlignment::MOCK, 0, 0, {0, 0}){};

std::pair<uint64_t, uint64_t> MockPhiloxGenerator::get_next_state() {
    return get_previous_state();
}

PhiloxOutput MockPhiloxGenerator::random() {
    PhiloxOutput result(ELEMENTS_PER_EXECUTION, 0);
    return result;
}

// ====== OpenvinoPhiloxGenerator functions ======
TensorflowPhiloxGenerator::TensorflowPhiloxGenerator(const uint64_t global_seed,
                                                     const uint64_t operator_seed,
                                                     const std::pair<uint64_t, uint64_t> previous_state)
    : PhiloxGenerator(op::PhiloxAlignment::TENSORFLOW,
                      global_seed,
                      previous_state.second > 0 ? previous_state.second : operator_seed,
                      previous_state),
      m_n64(previous_state.first),
      m_key64(global_seed),
      m_counter64(previous_state.second > 0 ? previous_state.second : operator_seed),
      m_n(split_high_low(m_n64)),
      m_key(split_high_low(m_key64)),
      m_counter(split_high_low(m_counter64)),
      m_total_generated_elements(0) {}

void TensorflowPhiloxGenerator::increment_key() {
    m_key.first += CRUSH_RESISTANCE_LOWER_VALUE;
    m_key.second += CRUSH_RESISTANCE_UPPER_VALUE;
}

void TensorflowPhiloxGenerator::execute_single_round() {
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

std::pair<uint64_t, uint64_t> TensorflowPhiloxGenerator::get_next_state() {
    // 256 is chosen to match Tensorflow's 'Reserve128Samples()' behavior
    const auto previous_state = get_previous_state();
    uint64_t skip_count = m_total_generated_elements * 256;
    uint64_t new_n = previous_state.first + skip_count;
    uint64_t new_op_seed = previous_state.second + (new_n < skip_count ? 1 : 0);

    return {new_n, new_op_seed};
}

PhiloxOutput TensorflowPhiloxGenerator::random() {
    PhiloxOutput result(ELEMENTS_PER_EXECUTION);

    // There are 10 rounds in the execution,
    // but the final one does not perform key increment
    for (uint16_t i = 0; i < SHUFFLE_ROUNDS - 1; ++i) {
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
    m_key = split_high_low(m_key64);
    m_counter = split_high_low(m_counter64);

    ++m_total_generated_elements;

    return result;
}

// ====== PytorchPhiloxGenerator functions ======
PytorchPhiloxGenerator::PytorchPhiloxGenerator(const uint64_t global_seed)
    : PhiloxGenerator(op::PhiloxAlignment::PYTORCH, global_seed, 0UL, {0UL, 0UL}),
      m_left(1),
      m_next(0) {
    m_mersenne_state[0] = global_seed & 0xffffffff;
    for (uint32_t j = 1; j < MERSENNE_STATE_N; ++j) {
        m_mersenne_state[j] = (1812433253 * (m_mersenne_state[j - 1] ^ (m_mersenne_state[j - 1] >> 30)) + j);
    }
}

std::pair<uint64_t, uint64_t> PytorchPhiloxGenerator::get_next_state() {
    return get_previous_state();
}

void PytorchPhiloxGenerator::next_mersenne_state() {
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

PhiloxOutput PytorchPhiloxGenerator::random() {
    PhiloxOutput result(ELEMENTS_PER_EXECUTION);

    for (size_t i = 0; i < ELEMENTS_PER_EXECUTION; ++i) {
        if (--m_left == 0) {
            next_mersenne_state();
        }

        result[i] = *(m_mersenne_state.data() + m_next++);
        result[i] ^= (result[i] >> 11);
        result[i] ^= (result[i] << 7) & 0x9d2c5680;
        result[i] ^= (result[i] << 15) & 0xefc60000;
        result[i] ^= (result[i] >> 18);
    }

    return result;
}

// ====== General selector function to construct a desired generator  ======

std::shared_ptr<PhiloxGenerator> make_philox_generator(uint64_t seed,
                                                       uint64_t seed2,
                                                       std::pair<uint64_t, uint64_t> prev_state,
                                                       size_t elem_count,
                                                       op::PhiloxAlignment alignment) {
    switch (alignment) {
    case op::PhiloxAlignment::TENSORFLOW:
        return std::make_shared<TensorflowPhiloxGenerator>(seed, seed2, prev_state);
    case op::PhiloxAlignment::PYTORCH:
        return std::make_shared<PytorchPhiloxGenerator>(seed);
    default:
        return std::make_shared<MockPhiloxGenerator>();
    }
}

}  // namespace philox
}  // namespace reference
}  // namespace ov
