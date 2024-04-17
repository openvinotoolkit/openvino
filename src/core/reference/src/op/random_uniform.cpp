// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/random_uniform.hpp"

#include <ctime>
#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/phillox_generator.hpp"

using PhilloxOutput = ov::reference::phillox::PhilloxOutput;
using PhilloxGenerator = ov::reference::phillox::PhilloxGenerator;
using PytorchPhilloxGenerator = ov::reference::phillox::PytorchPhilloxGenerator;
using OpenvinoPhilloxGenerator = ov::reference::phillox::OpenvinoPhilloxGenerator;
using TensorflowPhilloxGenerator = ov::reference::phillox::TensorflowPhilloxGenerator;

namespace ov {
namespace reference {
namespace {

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

// Helper function for converting uint32 values to float32. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float uint32_to_float(uint32_t x) {
    // float32 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(23 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 23 right bits from generated uint32 random value.

    convert_types out_val = {(static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu)};
    return out_val.f - 1.0f;
}

float uint32_to_float_pytorch(uint32_t x) {
    // float32 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(23 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 23 right bits from generated uint32 random value.
    auto MASK = static_cast<uint32_t>((static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    auto DIVISOR = static_cast<float>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits);
    convert_types out_val = {(static_cast<uint32_t>(127) << 23) | (x & 0x7fffffu)};
    float ret = (x & MASK) * DIVISOR;
    return ret;
}

// Helper function for converting uint32 values to float16.Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float16 uint32_to_float16(uint32_t x) {
    // float16 is formatted as follows: sign(1 bit) exponent(5 bits) mantissa(10 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 15)
    // Here we set the following values:
    // sign = 0
    // exponent = 15, for obtaining a zero exponent.
    // mantissa = 10 right bits from generated uint32 random value.

    uint16_t x_uint16 = static_cast<uint16_t>(x);
    convert_types out_val = {(static_cast<uint16_t>(15) << 10) | (x_uint16 & 0x3ffu)};
    return out_val.f16 - static_cast<float16>(1);
}

// Helper function for converting uint32 values to double. Sets fractional part of
// floating double with bits from uint32 values. Resulting value is in interval [0,1).
double uint32_to_double(uint32_t x1, uint32_t x2) {
    // float64 is formatted as follows: sign(1 bit) exponent(11 bits) mantissa(52 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 1023)
    // Here we set the following values:
    // sign = 0
    // exponent = 1023, for obtaining a zero exponent.
    // mantissa = 52 right bits from two concatenated uint32 values from random integer generator.

    uint64_t significant = ((static_cast<uint64_t>(x1) & 0xfffffu) << 32) | static_cast<uint64_t>(x2);
    convert_types out_val = {((static_cast<uint64_t>(1023) << 52) | significant)};
    return out_val.d - 1.0;
}

// Helper function for converting uint32 values to bfloat16. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
bfloat16 uint32_to_bfloat16(uint32_t x) {
    // bfloat16 is formatted as follows: sign(1 bit) exponent(8 bits) mantissa(7 bits). The value is interpreted
    // The value is interpreted using following formula:
    // (-1)^sign * 1, mantissa * 2 ^ (exponent - 127)
    // Here we set the following values:
    // sign = 0
    // exponent = 127, for obtaining a zero exponent.
    // mantissa = 7 right bits from generated uint32 random value.

    uint16_t x_uint16 = static_cast<uint16_t>(x);
    convert_types out_val = {(static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu)};
    return out_val.bf16 - static_cast<bfloat16>(1);
}

// Converts uint32 values to destination type and normalizes to required range
template <typename T>
void convert_to_output_type(const std::vector<uint32_t>& res,
                            size_t step,
                            const element::Type& elem_type,
                            const char* min_val,
                            const char* max_val,
                            char* out,
                            size_t k,
                            size_t elem_count,
                            T (*convert_single_input)(uint32_t) = nullptr,
                            T (*convert_two_inputs)(uint32_t, uint32_t, T, T) = nullptr,
                            T (*mod_func)(uint32_t, T, T) = nullptr) {
    // Get min and max values
    T mn[1];
    T mx[1];
    memcpy(mn, min_val, elem_type.size());
    memcpy(mx, max_val, elem_type.size());

    std::vector<T> res_out_type(step);
    if (elem_type.size() > 4) {
        // Each element of resulting sequence is formed using two uint32 values
        for (size_t i = 0; i < step / 2; ++i) {
            res_out_type[i] = convert_two_inputs(res[2 * i], res[2 * i + 1], mn[0], mx[0]);
        }
    } else {
        // Each element of resulting sequence is formed using single uint32 value
        std::transform(res.data(),
                       res.data() + step,
                       res_out_type.data(),
                       [&mn, &mx, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (mx[0] - mn[0]) + mn[0];
                           } else {
                               return mod_func(elem, mn[0], mx[0]);
                           }
                       });
    }

    memcpy(out + k * elem_type.size(), res_out_type.data(), std::min(step, elem_count - k) * elem_type.size());
}

}  // namespace

// Implementation of RandomUniform that uses Philox algorithm as inner random unsigned integer generator.
std::pair<uint64_t, uint64_t> random_uniform(const uint64_t* out_shape,
                                             const char* min_val,
                                             const char* max_val,
                                             char* out,
                                             const Shape& out_shape_shape,
                                             const element::Type& elem_type,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state,
                                             PhilloxAlignment alignment) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    // Implementation in plugins may differ for this case.
    if (seed == 0 && seed2 == 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seed = std::rand();
    }

    // Calculate total element count for generation
    size_t shape_count = shape_size(out_shape_shape);
    size_t elem_count = 1;
    for (size_t i = 0; i < shape_count; i++) {
        elem_count *= out_shape[i];
    }

    std::shared_ptr<PhilloxGenerator> generator;
    switch (alignment) {
    case PhilloxAlignment::OPENVINO:
        // Openvino uses seeds as a {key, counter} pair
        // seed -> global_seed <-> key
        // seed2 -> operator_seed <-> counter
        generator = std::make_shared<OpenvinoPhilloxGenerator>(seed, seed2, prev_state);
        break;
    case PhilloxAlignment::TENSORFLOW:
        // Very similar algorithm
        generator = std::make_shared<TensorflowPhilloxGenerator>(seed, seed2, prev_state, elem_count);
        break;
    case PhilloxAlignment::PYTORCH:
        // Completely different algorithm that uses only a single seed
        generator = std::make_shared<PytorchPhilloxGenerator>(seed);
        break;
    default:
        OPENVINO_THROW("Unknown Phillox algorithm alignment option selected.");
    }

    const size_t step = generator->get_step(elem_type);
    for (size_t k = 0; k < elem_count; k += step) {
        // generate a set of random uint32 values using Philox algorithm
        PhilloxOutput result = generator->random();

        // convert values to corresponding output_type
        switch (elem_type) {
        case element::Type_t::f32: {
            convert_to_output_type<float>(result,
                                          step,
                                          elem_type,
                                          min_val,
                                          max_val,
                                          out,
                                          k,
                                          elem_count,
                                          uint32_to_float); // TODO CHANGE THIS
            break;
        }
        case element::Type_t::f16: {
            convert_to_output_type<float16>(result,
                                            step,
                                            elem_type,
                                            min_val,
                                            max_val,
                                            out,
                                            k,
                                            elem_count,
                                            uint32_to_float16);
            break;
        }
        case element::Type_t::bf16: {
            convert_to_output_type<bfloat16>(result,
                                             step,
                                             elem_type,
                                             min_val,
                                             max_val,
                                             out,
                                             k,
                                             elem_count,
                                             uint32_to_bfloat16);
            break;
        }
        case element::Type_t::f64: {
            convert_to_output_type<double>(result,
                                           step,
                                           elem_type,
                                           min_val,
                                           max_val,
                                           out,
                                           k,
                                           elem_count,
                                           nullptr,
                                           [](uint32_t a, uint32_t b, double mn, double mx) {
                                               return uint32_to_double(a, b) * (mx - mn) + mn;
                                           });
            break;
        }
        case element::Type_t::i32: {
            convert_to_output_type<int>(result,
                                        step,
                                        elem_type,
                                        min_val,
                                        max_val,
                                        out,
                                        k,
                                        elem_count,
                                        nullptr,
                                        nullptr,
                                        [](uint32_t x, int mn, int mx) {
                                            return static_cast<int>(x % (mx - mn) + mn);
                                        });
            break;
        }
        case element::Type_t::i64: {
            convert_to_output_type<int64_t>(result,
                                            step,
                                            elem_type,
                                            min_val,
                                            max_val,
                                            out,
                                            k,
                                            elem_count,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                            });
            break;
        }
        default:
            OPENVINO_THROW("Unsupported type of RandomUniform: ", elem_type.to_string());
        }
    }

    return generator->get_next_state();
}

}  // namespace reference
}  // namespace ov
