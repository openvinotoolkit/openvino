// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ctime>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace reference {

// Helper struct for converting between types
struct convert_types {
    union {
        uint64_t ui64;
        double d;
        float f;
        float16 f16;
        bfloat16 bf16;
    };
};

std::pair<uint64_t, uint64_t> random_uniform(const uint64_t* out_shape,
                                             const char* min_val,
                                             const char* max_val,
                                             char* out,
                                             const Shape& out_shape_shape,
                                             const element::Type& elem_type,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state);

// Following const values are taken from the original paper:
// https://www.thesalmons.org/john/random123/papers/random123sc11.pdf
const uint32_t crush_resistance_const_lower_value = 0x9E3779B9;
const uint32_t crush_resistance_const_upper_value = 0xBB67AE85;
const uint64_t statistic_maximizing_multiplier_n = 0xD2511F53;
const uint64_t statistic_maximizing_multiplier_counter = 0xCD9E8D57;
const size_t rounds_number = 10;

// Determines how many sequence elements of RNG sequence are skipped between runs.
// Can be any positive value, 256 is chosen for parity with Tensorflow.
const uint64_t skip_const = 256;

}  // namespace reference
}  // namespace ov
