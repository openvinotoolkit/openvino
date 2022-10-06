// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/random_uniform.hpp"

#include <ctime>

#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
namespace {

// Splits uint64 value into two uint32 values with right and left part of original value.
std::pair<uint32_t, uint32_t> split_high_low(uint64_t value) {
    uint32_t low = static_cast<uint32_t>(value);
    uint32_t high = static_cast<uint32_t>(value >> 32);
    return {low, high};
}

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low) {
    return (static_cast<uint64_t>(high) << 32) + low;
}

// Runs single "round" of Philox algorithm.
void calculate_round(uint64_t key, uint64_t& counter, uint64_t& n) {
    // Split key, counter and n into two uint32 values.
    auto counter_lr = split_high_low(counter);
    auto key_lr = split_high_low(key);
    auto n_lr = split_high_low(n);

    // Each round performs following updating for n and counter:
    // left uint32 part = mullo(R, M)
    // right uint32 part  = mulhi(R, M) xor k xor L
    // mulhi(a, b) = floor((a * b) / 2^32)
    // mullo(a, b) = (a * b) mod 2^32,
    // where M - statistic_maximizing_multiplier const
    auto prod0 = split_high_low(statistic_maximizing_multiplier_n * n_lr.first);
    auto prod1 = split_high_low(statistic_maximizing_multiplier_counter * counter_lr.first);
    n_lr.first = prod1.second ^ n_lr.second ^ key_lr.first;
    n_lr.second = prod1.first;
    counter_lr.first = prod0.second ^ counter_lr.second ^ key_lr.second;
    counter_lr.second = prod0.first;

    // Unite counter and n into uint64 values.
    counter = unite_high_low(counter_lr.second, counter_lr.first);
    n = unite_high_low(n_lr.second, n_lr.first);
}

// Increases key value.
void raise_key(uint64_t& key) {
    auto key_lr = split_high_low(key);
    key_lr.first += crush_resistance_const_lower_value;
    key_lr.second += crush_resistance_const_upper_value;
    key = unite_high_low(key_lr.second, key_lr.first);
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

// Runs Philox algorithm.
void run_philox(uint64_t key, uint64_t counter, uint64_t n, size_t n_rounds, std::vector<uint32_t>& res) {
    for (size_t i = 0; i < n_rounds; i++) {
        calculate_round(key, counter, n);
        if (i < n_rounds - 1)
            raise_key(key);
    }
    auto res1 = split_high_low(n);
    auto res2 = split_high_low(counter);
    res[0] = res1.first;
    res[1] = res1.second;
    res[2] = res2.first;
    res[3] = res2.second;
}

// Converts uint32 values to destination type and normalizes to required range
template <typename T>
void convert_to_output_type(const std::vector<uint32_t>& res,
                            size_t step,
                            const ngraph::element::Type& elem_type,
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
        res_out_type[0] = convert_two_inputs(res[0], res[1], mn[0], mx[0]);
        res_out_type[1] = convert_two_inputs(res[2], res[3], mn[0], mx[0]);
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
                                             const ngraph::element::Type& elem_type,
                                             uint64_t seed,
                                             uint64_t seed2,
                                             std::pair<uint64_t, uint64_t> prev_state) {
    // When both seed values are equal to zero RandomUniform should generate non-deterministic sequence.
    // Implementation in plugins may differ for this case.
    if (seed == 0 && seed2 == 0) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        seed = std::rand();
    }

    // Get previous counter state
    uint64_t n_state = prev_state.first;
    uint64_t counter_state = prev_state.second;

    // Initialize Philox key and counters
    uint64_t key = seed;
    uint64_t counter = counter_state > 0 ? counter_state : seed2;
    uint64_t n = n_state;

    // Calculate total element count for generation
    size_t shape_count = shape_size(out_shape_shape);
    size_t elem_count = 1;
    for (size_t i = 0; i < shape_count; i++) {
        elem_count *= out_shape[i];
    }

    // Philox algorithm returns 4 elements of RNG sequence per each invocation
    const size_t philox_output_size = 4;

    // Each run of Philox algorithm generates 4 uint32 values.
    // If output_type is int32, f32, bf16, or f16 each value is converted to
    // corresponding type so we have 4 result values. For f64 and i64 we use
    // a pair of values for conversion, so we have 2 result values.
    // Step indicates how many values we generate in one iteration.
    const size_t step = elem_type.size() > 4 ? 2 : 4;

    for (size_t k = 0; k < elem_count; k += step) {
        // generate 4 random uint32 values using Philox algorithm
        std::vector<uint32_t> res(philox_output_size);
        run_philox(key, counter, n, rounds_number, res);

        // convert values to corresponding output_type
        switch (elem_type) {
        case ngraph::element::Type_t::f32: {
            convert_to_output_type<float>(res, step, elem_type, min_val, max_val, out, k, elem_count, uint32_to_float);
            break;
        }
        case ngraph::element::Type_t::f16: {
            convert_to_output_type<float16>(res,
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
        case ngraph::element::Type_t::bf16: {
            convert_to_output_type<bfloat16>(res,
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
        case ngraph::element::Type_t::f64: {
            convert_to_output_type<double>(res,
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
        case ngraph::element::Type_t::i32: {
            convert_to_output_type<int>(res,
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
        case ngraph::element::Type_t::i64: {
            convert_to_output_type<int64_t>(res,
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
            throw ngraph_error("Unsupported type of RandomUniform: " + elem_type.get_type_name());
        }
        if (++n == 0)
            ++counter;
    }

    // Calculate counter values for next RandomUniform run
    uint64_t skip_count = elem_count * skip_const;
    n_state += skip_count;
    if (n_state < skip_count)
        counter_state++;

    return {n_state, counter_state};
}

}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
