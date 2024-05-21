// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/phillox_generator.hpp"

using PhilloxAlignment = ov::op::PhilloxAlignment;
using PhilloxOutput = ov::reference::phillox::PhilloxOutput;

namespace ov {

namespace reference {

namespace phillox {

namespace {

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

template <typename T>
struct PytorchDistributionAccumulatorType {};
template <>
struct PytorchDistributionAccumulatorType<ov::bfloat16> {
    using type = float;
};
template <>
struct PytorchDistributionAccumulatorType<ov::float16> {
    using type = float;
};
template <>
struct PytorchDistributionAccumulatorType<float> {
    using type = float;
};
template <>
struct PytorchDistributionAccumulatorType<double> {
    using type = double;
};

template <typename T>
using PytorchAccumulatorType = typename PytorchDistributionAccumulatorType<T>::type;

template <typename T>
T uint32_to_T_type_float(uint32_t x) {
    auto MASK = static_cast<uint32_t>((static_cast<uint64_t>(1) << std::numeric_limits<float>::digits) - 1);
    auto DIVISOR = static_cast<float>(1) / (static_cast<uint64_t>(1) << std::numeric_limits<float>::digits);
    PytorchAccumulatorType<T> ret = (x & MASK) * DIVISOR;
    return static_cast<T>(ret);
}

template <typename T>
T uint32_to_T_type_int(uint32_t x, uint64_t range, int64_t base) {
    return static_cast<T>(static_cast<int64_t>((x % range) + base));
}

}  // namespace

class PhilloxConverter {
public:
    virtual void convert(PhilloxOutput result, size_t k) = 0;

protected:
    PhilloxConverter(char* out,
                     const size_t step,
                     const element::Type& elem_type,
                     const size_t elem_count,
                     const char* min_val,
                     const char* max_val)
        : m_out(out),
          m_step(step),
          m_elem_count(elem_count),
          m_elem_type(elem_type),
          m_min_val(min_val),
          m_max_val(max_val) {}

    char* m_out;
    const size_t m_step;
    const size_t m_elem_count;
    const element::Type m_elem_type;
    const char* m_min_val;
    const char* m_max_val;
};

class TensorflowPhilloxConverter : public PhilloxConverter {
public:
    TensorflowPhilloxConverter(char* out,
                               const size_t step,
                               const element::Type& elem_type,
                               const size_t elem_count,
                               const char* min_val,
                               const char* max_val)
        : PhilloxConverter(out, step, elem_type, elem_count, min_val, max_val) {}

    void convert(PhilloxOutput result, size_t k) override {
        // convert values to corresponding output_type
        switch (m_elem_type) {
        case element::Type_t::f32: {
            convert_to_output_type<float>(result,
                                          m_step,
                                          m_elem_type,
                                          m_min_val,
                                          m_max_val,
                                          m_out,
                                          k,
                                          m_elem_count,
                                          uint32_to_float);
            break;
        }
        case element::Type_t::f16: {
            convert_to_output_type<float16>(result,
                                            m_step,
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            k,
                                            m_elem_count,
                                            uint32_to_float16);
            break;
        }
        case element::Type_t::bf16: {
            convert_to_output_type<bfloat16>(result,
                                             m_step,
                                             m_elem_type,
                                             m_min_val,
                                             m_max_val,
                                             m_out,
                                             k,
                                             m_elem_count,
                                             uint32_to_bfloat16);
            break;
        }
        case element::Type_t::f64: {
            convert_to_output_type<double>(result,
                                           m_step,
                                           m_elem_type,
                                           m_min_val,
                                           m_max_val,
                                           m_out,
                                           k,
                                           m_elem_count,
                                           nullptr,
                                           [](uint32_t a, uint32_t b, double mn, double mx) {
                                               return uint32_to_double(a, b) * (mx - mn) + mn;
                                           });
            break;
        }
        case element::Type_t::i32: {
            convert_to_output_type<int>(result,
                                        m_step,
                                        m_elem_type,
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        k,
                                        m_elem_count,
                                        nullptr,
                                        nullptr,
                                        [](uint32_t x, int mn, int mx) {
                                            return static_cast<int>(x % (mx - mn) + mn);
                                        });
            break;
        }
        case element::Type_t::i64: {
            convert_to_output_type<int64_t>(result,
                                            m_step,
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            k,
                                            m_elem_count,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                            });
            break;
        }
        default:
            OPENVINO_THROW("Unsupported type of RandomUniform: ", m_elem_type.to_string());
        }
    }
};

class PyTorchPhilloxConverter : public PhilloxConverter {
public:
    PyTorchPhilloxConverter(char* out,
                            const size_t step,
                            const element::Type& elem_type,
                            const size_t elem_count,
                            const char* min_val,
                            const char* max_val)
        : PhilloxConverter(out, step, elem_type, elem_count, min_val, max_val) {}

    void convert(PhilloxOutput result, size_t k) override {
        // convert values to corresponding output_type
        switch (m_elem_type) {
        case element::Type_t::f32: {
            convert_to_output_type<float>(result,
                                          m_step,
                                          m_elem_type,
                                          m_min_val,
                                          m_max_val,
                                          m_out,
                                          k,
                                          m_elem_count,
                                          uint32_to_T_type_float<float>);
            break;
        }
        case element::Type_t::f16: {
            convert_to_output_type<float16>(result,
                                            m_step,
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            k,
                                            m_elem_count,
                                            uint32_to_T_type_float<ov::float16>);
            break;
        }
        case element::Type_t::bf16: {
            convert_to_output_type<bfloat16>(result,
                                             m_step,
                                             m_elem_type,
                                             m_min_val,
                                             m_max_val,
                                             m_out,
                                             k,
                                             m_elem_count,
                                             uint32_to_T_type_float<ov::bfloat16>);
            break;
        }
        case element::Type_t::f64: {
            convert_to_output_type<double>(result,
                                           m_step,
                                           m_elem_type,
                                           m_min_val,
                                           m_max_val,
                                           m_out,
                                           k,
                                           m_elem_count,
                                           uint32_to_T_type_float<double>);
            break;
        }
        case element::Type_t::i32: {
            convert_to_output_type<int>(result,
                                        m_step,
                                        m_elem_type,
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        k,
                                        m_elem_count,
                                        nullptr,
                                        nullptr,
                                        [](uint32_t x, int mn, int mx) {
                                            return static_cast<int>(x % (mx - mn) + mn);
                                        });
            break;
        }
        case element::Type_t::i64: {
            convert_to_output_type<int64_t>(result,
                                            m_step,
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            k,
                                            m_elem_count,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                return static_cast<int64_t>(unite_high_low(b, a) % (mx - mn) + mn);
                                            });
            break;
        }
        default:
            OPENVINO_THROW("Unsupported type of RandomUniform: ", m_elem_type.to_string());
        }
    }
};

static std::shared_ptr<PhilloxConverter> make_phillox_converter(char* out,
                                                                const element::Type& elem_type,
                                                                const char* min_val,
                                                                const char* max_val,
                                                                const size_t elem_count,
                                                                const std::shared_ptr<PhilloxGenerator> generator) {
    switch (generator->get_alignment()) {
    case PhilloxAlignment::OPENVINO:
    case PhilloxAlignment::TENSORFLOW:
        // Exactly the same conversion for Openvino and Tensorflow
        return std::make_shared<TensorflowPhilloxConverter>(out,
                                                            generator->get_step(elem_type),
                                                            elem_type,
                                                            elem_count,
                                                            min_val,
                                                            max_val);
    case PhilloxAlignment::PYTORCH:
        // Different conversion
        return std::make_shared<PyTorchPhilloxConverter>(out,
                                                         generator->get_step(elem_type),
                                                         elem_type,
                                                         elem_count,
                                                         min_val,
                                                         max_val);
    default:
        OPENVINO_THROW("Unknown Phillox algorithm alignment option selected.");
    }
}

}  // namespace phillox
}  // namespace reference
}  // namespace ov
