// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"

using PhilloxAlignment = ov::op::PhilloxAlignment;

namespace ov {

namespace reference {

namespace phillox {

typedef std::vector<uint32_t> PhilloxOutput;

/// \brief Generator of random numbers based on the Phillox algorithm.
///        Abstract base class for various specializations
///        used to match outputs based on input seed(s)
///        for supported frameworks
class PhilloxGenerator {
public:
    /// \brief Constructor for the PhilloxGenerator class.
    /// \param previous_state The previous state (if any) of the algorithm
    /// \param global_seed The global seed for the Phillox algorithm
    /// \param operator_seed The operator seed for the Phillox algorithm
    /// \param output_size The amount of elements generated per function call.

    PhilloxGenerator() = delete;

    /// @brief Get a set of random 32-bit unsigned integers based on the seed(s).
    /// @return A vector with a random set of 32-bit unsigned integers.
    virtual PhilloxOutput random() = 0;

    virtual size_t get_step(const element::Type& elem_type) const = 0;

    virtual std::pair<uint64_t, uint64_t> get_next_state() = 0;

    /// \brief Getter for the global seed
    /// \return The global seed for the Phillox algorithm
    uint64_t get_global_seed() const {
        return m_global_seed;
    }

    /// \brief Getter for the operator seed
    /// \return The operator seed for the Phillox algorithm
    uint64_t get_operator_seed() const {
        return m_operator_seed;
    }

    std::pair<uint64_t, uint64_t> get_previous_state() const {
        return m_previous_state;
    }

    size_t get_output_size() const {
        return m_output_size;
    }

    PhilloxAlignment get_alignment() const {
        return m_alignment;
    }

    /// \brief Setter for the global seed
    /// \param global_seed The new global seed for the Phillox algorithm
    void set_global_seed(const uint64_t global_seed) {
        m_global_seed = global_seed;
    }

    /// \brief Setter for the operator seed
    /// \param op_seed The new operator seed for the Phillox algorithm
    void set_operator_seed(const uint64_t op_seed) {
        m_operator_seed = op_seed;
    }

    void set_previous_state(const std::pair<uint64_t, uint64_t> previous_state) {
        m_previous_state = previous_state;
    }

protected:
    /// Note: Depending on the algorithm, one or both seeds might be used.
    PhilloxGenerator(const uint64_t global_seed,
                     const uint64_t operator_seed,
                     const size_t output_size,
                     const std::pair<uint64_t, uint64_t> previous_state,
                     const PhilloxAlignment alignment)
        : m_global_seed(global_seed),
          m_operator_seed(operator_seed),
          m_previous_state(previous_state),
          m_output_size(output_size),
          m_alignment(alignment) {}

    uint64_t m_global_seed = 0;
    uint64_t m_operator_seed = 0;
    std::pair<uint64_t, uint64_t> m_previous_state = {0, 0};

private:
    size_t m_output_size = 0;
    const PhilloxAlignment m_alignment;
};

/// \brief OpenVINO specialization of the PhilloxGenerator class.
class OpenvinoPhilloxGenerator : public PhilloxGenerator {
public:
    /// \brief Constructor for the OpenvinoPhilloxGenerator class.
    /// \param global_seed The global seed for the Phillox algorithm
    /// \param operator_seed The operator seed for the Phillox algorithm
    OpenvinoPhilloxGenerator(const uint64_t global_seed,
                             const uint64_t operator_seed,
                             const std::pair<uint64_t, uint64_t> previous_state)
        : PhilloxGenerator(global_seed, operator_seed, 4, previous_state, PhilloxAlignment::OPENVINO) {
        // Initialize Phillox constants: key, counter, n
        if (previous_state.second > 0) {
            set_operator_seed(previous_state.second);
        }
        m_n = previous_state.first;
    }

    /// @brief Get a set of 4 random 32-bit unsigned integers based on the seeds.
    /// @return A structure with a random set of 32-bit unsigned integers.
    PhilloxOutput random() override;

    size_t get_step(const element::Type& elem_type) const override {
        // Each run of Philox algorithm generates 4 uint32 values.
        // If output_type is int32, f32, bf16, or f16 each value is converted to
        // corresponding type so we have 4 result values. For f64 and i64 we use
        // a pair of values for conversion, so we have 2 result values.
        // Step indicates how many values we generate in one iteration.
        return elem_type.size() > 4 ? 2 : 4;
    }

    std::pair<uint64_t, uint64_t> get_next_state() override {
        uint64_t skip_count = m_generated_elements_count * m_skip_const;
        uint64_t new_n = m_previous_state.first + skip_count;
        if (new_n < skip_count) {
            return {new_n, m_previous_state.second + 1};
        }
        return {new_n, m_previous_state.second};
    }

private:
    uint64_t m_n = 0;
    size_t m_generated_elements_count = 0;

    void execute_single_round();
    void increment_key();

    // Following const values are taken from the original paper:
    // https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
    const uint32_t m_crush_resistance_const_lower_value = 0x9E3779B9;
    const uint32_t m_crush_resistance_const_upper_value = 0xBB67AE85;
    const uint64_t m_statistic_maximizing_multiplier_n = 0xD2511F53;
    const uint64_t m_statistic_maximizing_multiplier_counter = 0xCD9E8D57;

    // Determines how many sequence elements of RNG sequence are skipped between runs.
    // Can be any positive value, 256 is chosen for parity with Tensorflow.
    const uint64_t m_skip_const = 256;
};

static constexpr int _PYTORCH_MERSENNE_STATE_N = 624;
static constexpr int _PYTORCH_MERSENNE_STATE_M = 397;

/// \brief PyTorch specialization of the PhilloxGenerator class.
class PytorchPhilloxGenerator : public PhilloxGenerator {
public:
    /// \brief Constructor for the PytorchPhilloxGenerator class.
    /// \param global_seed The operator seed for the Phillox algorithm
    /// Note: PyTorch specialization of Phillox algorithm does not use operator seed.
    PytorchPhilloxGenerator(const uint64_t global_seed)
        : PhilloxGenerator(global_seed, 0, 2, {0, 0}, PhilloxAlignment::PYTORCH) {
        m_seeded = true;
        m_state[0] = global_seed & 0xffffffff;
        m_left = 1;
        m_next = 0;
        for (size_t j = 1; j < _PYTORCH_MERSENNE_STATE_N; ++j) {
            m_state[j] = (1812433253 * (m_state[j - 1] ^ (m_state[j - 1] >> 30)) + j);
        }
    }

    size_t get_step(const element::Type& elem_type) const override {
        return elem_type.size() > 4 ? 1 : 2;
    }

    std::pair<uint64_t, uint64_t> get_next_state() override {
        return m_previous_state;
    }

    /// @brief Get a set of random 32-bit unsigned integers based on the seed(s).
    /// @return A structure with a random set of 32-bit unsigned integers and their count.
    PhilloxOutput random() override;

private:
    void next_state();

    int m_left;
    bool m_seeded;
    uint32_t m_next;
    std::array<uint32_t, _PYTORCH_MERSENNE_STATE_N> m_state;
};

static constexpr uint32_t _TENSORFLOW_kPhiloxW32A = 0x9E3779B9;
static constexpr uint32_t _TENSORFLOW_kPhiloxW32B = 0xBB67AE85;
static constexpr uint32_t _TENSORFLOW_kPhiloxM4x32A = 0xD2511F53;
static constexpr uint32_t _TENSORFLOW_kPhiloxM4x32B = 0xCD9E8D57;

/// \brief Tensorflow specialization of the PhilloxGenerator class.
class TensorflowPhilloxGenerator : public PhilloxGenerator {
public:
    /// \brief Constructor for the TensorflowPhilloxGenerator class.
    /// \param global_seed The global seed for the Phillox algorithm
    /// \param operator_seed The operator seed for the Phillox algorithm
    TensorflowPhilloxGenerator(const uint64_t global_seed,
                               const uint64_t operator_seed,
                               const std::pair<uint64_t, uint64_t> previous_state,
                               uint64_t elements_count)
        : PhilloxGenerator(global_seed, operator_seed, 4, previous_state, PhilloxAlignment::TENSORFLOW) {
        m_key[0] = static_cast<uint32_t>(global_seed);
        m_key[1] = static_cast<uint32_t>(global_seed >> 32);
        m_counter[0] = static_cast<uint32_t>(previous_state.first);
        m_counter[1] = static_cast<uint32_t>(previous_state.first >> 32);

        if (previous_state.second == 0) {
            m_counter[2] = static_cast<uint32_t>(operator_seed);
            m_counter[3] = static_cast<uint32_t>(operator_seed >> 32);
        } else {
            m_counter[2] = static_cast<uint32_t>(previous_state.second);
            m_counter[3] = static_cast<uint32_t>(previous_state.second >> 32);
        }

        // skip(elements_count);
    }

    size_t get_step(const element::Type& elem_type) const override {
        return elem_type.size() > 4 ? 2 : 4;
    }

    std::pair<uint64_t, uint64_t> get_next_state() override {
        return {static_cast<uint64_t>(m_counter[1]) + m_counter[0], static_cast<uint64_t>(m_counter[3]) + m_counter[2]};
    }

    /// @brief Get a set of random 32-bit unsigned integers based on the seed(s).
    /// @return A structure with a random set of 32-bit unsigned integers and their count.
    PhilloxOutput random() override;

private:
    void skip(uint64_t count);
    void skip_one();
    void skip_256();
    void raise_key();
    void compute_single_round();

    std::array<uint32_t, 2> m_key;
    std::array<uint32_t, 4> m_counter;
};

static std::shared_ptr<PhilloxGenerator> make_phillox_generator(uint64_t seed,
                                                                uint64_t seed2,
                                                                std::pair<uint64_t, uint64_t> prev_state,
                                                                size_t elem_count,
                                                                PhilloxAlignment alignment) {
    switch (alignment) {
    case PhilloxAlignment::OPENVINO:
        // Openvino uses seeds as a {key, counter} pair
        // seed -> global_seed <-> key
        // seed2 -> operator_seed <-> counter
        return std::make_shared<OpenvinoPhilloxGenerator>(seed, seed2, prev_state);
    case PhilloxAlignment::TENSORFLOW:
        // Very similar algorithm
        return std::make_shared<TensorflowPhilloxGenerator>(seed, seed2, prev_state, elem_count);
    case PhilloxAlignment::PYTORCH:
        // Completely different algorithm that uses only a single seed
        return std::make_shared<PytorchPhilloxGenerator>(seed);
    default:
        OPENVINO_THROW("Unknown Phillox algorithm alignment option selected.");
    }
}

}  // namespace phillox

}  // namespace reference

}  // namespace ov
