// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cstdint>
#include <utility>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"

using PhilloxAlignment = ov::op::PhilloxAlignment;
namespace ov {
namespace reference {
namespace phillox {

typedef std::vector<uint32_t> PhilloxOutput;

// All generators are aligned to output exactly 4 uint32_t values.
// This value is shared between generators and converters.
static constexpr size_t ELEMENTS_PER_EXECUTION = 4;

/// \brief Generator of random numbers based on the Phillox algorithm.
///        Abstract base class for various specializations
///        used to match outputs based on input seed(s)
///        for supported frameworks
class PhilloxGenerator {
public:
    PhilloxGenerator() = delete;

    virtual ~PhilloxGenerator(){};

    /// @brief Get a set of 4 random 32-bit unsigned integers based on the seed(s).
    /// @return A vector with a random set of 4 32-bit unsigned integers.
    virtual PhilloxOutput random() = 0;

    /// \brief Returns the modified state to feed to the next execution.
    /// @return A pair of uint64s that represent the output state to be fed to the next generator execution.
    virtual std::pair<uint64_t, uint64_t> get_next_state() = 0;

    // ========================================================

    /// \brief Returns the global seed of the generator.
    uint64_t get_global_seed() const;

    /// \brief Returns the operator seed of the generator.
    uint64_t get_operator_seed() const;

    /// \brief Returns the alignment mode of the generator.
    PhilloxAlignment get_alignment() const;

    /// \brief Returns the input (previous execution) state of the generator.
    std::pair<uint64_t, uint64_t> get_previous_state() const;

    /// \brief Setter for the global seed
    /// \param global_seed The new global seed for the Phillox algorithm
    void set_global_seed(const uint64_t global_seed);

    /// \brief Setter for the operator seed
    /// \param op_seed The new operator seed for the Phillox algorithm
    void set_operator_seed(const uint64_t op_seed);

protected:
    /// \brief Constructor for the PhilloxGenerator class.
    /// \param alignment The alignment mode of the generator.
    /// \param global_seed The global seed for the Phillox algorithm.
    /// \param operator_seed The operator seed for the Phillox algorithm.
    /// \param previous_state The previous state (if any) of the algorithm.
    /// \param generated_elements_count The amount of elements generated per execution of random().
    PhilloxGenerator(const PhilloxAlignment alignment,
                     const uint64_t global_seed,
                     const uint64_t operator_seed,
                     const std::pair<uint64_t, uint64_t> previous_state);

private:
    const PhilloxAlignment m_alignment;
    uint64_t m_global_seed;
    uint64_t m_operator_seed;

protected:
    const std::pair<uint64_t, uint64_t> m_previous_state;
};

/// \brief Mock specialization of the PhilloxGenerator class (in case of unknown alignment).
class MockPhilloxGenerator : public PhilloxGenerator {
public:
    MockPhilloxGenerator();

    /// @brief Get a set of 4 32-bit unsigned integers (zeros).
    /// @return A structure with a set of 32-bit zeros.
    PhilloxOutput random() override;

    /// \brief Returns the modified state to feed to the next execution.
    /// @return A pair of uint64s that represent the output state to be fed to the next generator execution.
    std::pair<uint64_t, uint64_t> get_next_state() override;
};

/// \brief OpenVINO specialization of the PhilloxGenerator class.
class TensorflowPhilloxGenerator : public PhilloxGenerator {
public:
    TensorflowPhilloxGenerator() = delete;

    /// \brief Constructor for the TensorflowPhilloxGenerator class.
    /// \param global_seed The global seed for the Phillox algorithm
    /// \param operator_seed The operator seed for the Phillox algorithm
    /// \param previous_state The state returned from the previous execution of the generator
    TensorflowPhilloxGenerator(const uint64_t global_seed,
                               const uint64_t operator_seed,
                               const std::pair<uint64_t, uint64_t> previous_state);

    /// @brief Get a set of 4 random 32-bit unsigned integers based on the seeds.
    /// @return A structure with a random set of 32-bit unsigned integers.
    PhilloxOutput random() override;

    /// \brief Returns the modified state to feed to the next execution.
    /// @return A pair of uint64s that represent the output state to be fed to the next generator execution.
    std::pair<uint64_t, uint64_t> get_next_state() override;

private:
    // ====== Mersenne Twister paper variables ======
    // Following const values are taken from the original paper:
    // https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
    //
    // Values for OpenVINO, Tensorflow
    static constexpr uint32_t CRUSH_RESISTANCE_LOWER_VALUE = 0x9E3779B9;
    static constexpr uint32_t CRUSH_RESISTANCE_UPPER_VALUE = 0xBB67AE85;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_N = 0xD2511F53;
    static constexpr uint64_t STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER = 0xCD9E8D57;
    static constexpr uint16_t SHUFFLE_ROUNDS = 10;

    void execute_single_round();
    void increment_key();

    uint64_t m_n64;
    uint64_t m_key64;
    uint64_t m_counter64;
    std::pair<uint32_t, uint32_t> m_n;
    std::pair<uint32_t, uint32_t> m_key;
    std::pair<uint32_t, uint32_t> m_counter;

    size_t m_total_generated_elements;
};

/// \brief PyTorch specialization of the PhilloxGenerator class.
class PytorchPhilloxGenerator : public PhilloxGenerator {
public:
    PytorchPhilloxGenerator() = delete;

    /// \brief Constructor for the PytorchPhilloxGenerator class.
    /// \param global_seed The operator seed for the Phillox algorithm
    /// \note This version of the Phillox algorithm does NOT use operator seed, and.
    /// does not support the use of the previous/next state.
    PytorchPhilloxGenerator(const uint64_t global_seed);

    /// @brief Get a set of random 32-bit unsigned integers based on the seed(s).
    /// @return A structure with a random set of 32-bit unsigned integers and their count.
    PhilloxOutput random() override;

    /// \brief Returns the INPUT state to feed to the next execution, as the algorithm does not use this to compute
    /// results.
    std::pair<uint64_t, uint64_t> get_next_state() override;

private:
    // Values for PyTorch
    static constexpr int MERSENNE_STATE_N = 624;
    static constexpr int MERSENNE_STATE_M = 397;

    void next_mersenne_state();

    uint32_t m_left;
    uint32_t m_next;
    std::array<uint32_t, MERSENNE_STATE_N> m_mersenne_state;
};

/// \brief Constructs and returns a shared pointer to the generator chosen by alignment.
std::shared_ptr<PhilloxGenerator> make_phillox_generator(const uint64_t seed,
                                                         const uint64_t seed2,
                                                         const std::pair<uint64_t, uint64_t> prev_state,
                                                         const size_t elem_count,
                                                         const PhilloxAlignment alignment);

}  // namespace phillox

}  // namespace reference

}  // namespace ov
