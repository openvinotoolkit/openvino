// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/phillox_converter.hpp"

namespace ov {

namespace reference {

namespace phillox {

// Utils functions from OpenVINO, Tensorflow, PyTorch
namespace {

// Helper struct for converting between types in OpenVINO
struct convert_types {
    union {
        uint64_t ui64;
        double d;
        float f;
        float16 f16;
        bfloat16 bf16;
    };
};

// Helper struct for converting between types in PyTorch
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

    auto x_uint16 = static_cast<uint16_t>(x);
    convert_types out_val = {(static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu)};
    return out_val.bf16 - static_cast<bfloat16>(1);
}

// Helper function for converting uint32 values to any T type float, using PyTorch
// conversion method. Sets fractional part of floating value with bits from
// uint32 value. Resulting value is in interval [0,1).
template <typename T, typename V>
PytorchAccumulatorType<T> uint32_to_T_type_float(V x) {
    const auto mask = static_cast<V>((1UL << std::numeric_limits<T>::digits) - 1);
    const auto divisor = static_cast<PytorchAccumulatorType<T>>(1) / (1UL << std::numeric_limits<T>::digits);
    PytorchAccumulatorType<T> ret = (x & mask) * divisor;
    return ret;
}

float uint32_to_bfloat16_pytorch(uint32_t x) {
    const auto mask = static_cast<uint32_t>((1UL << 8) - 1);
    const auto divisor = static_cast<PytorchAccumulatorType<ov::bfloat16>>(1) / (1UL << 8);
    PytorchAccumulatorType<ov::bfloat16> ret = (x & mask) * divisor;
    return ret;
}

// Converts uint32 values to destination type and normalizes to required range.
template <typename T>
void convert_to_output_type(const std::vector<uint32_t>& res,
                            size_t uints_generated_per_exec,
                            size_t output_type_generated_values,
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

    std::vector<T> res_out_type(uints_generated_per_exec);
    // Either 2 or 4 (2 for 64 bit values, 4 otherwise)
    if (output_type_generated_values == 2) {
        // Each element of resulting sequence is formed using two uint32 values,
        // therefore only half of the res_out_type array is used.
        for (size_t i = 0; i < uints_generated_per_exec / 2; ++i) {
            res_out_type[i] = convert_two_inputs(res[2 * i], res[2 * i + 1], mn[0], mx[0]);
        }
    } else {
        // Each element of resulting sequence is formed using single uint32 value
        std::transform(res.data(),
                       res.data() + uints_generated_per_exec,
                       res_out_type.data(),
                       [&mn, &mx, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (mx[0] - mn[0]) + mn[0];
                           } else {
                               return mod_func(elem, mn[0], mx[0]);
                           }
                       });
    }

    memcpy(out + k * elem_type.size(),
           res_out_type.data(),
           std::min(uints_generated_per_exec, elem_count - k) * elem_type.size());
}

}  // namespace

// ======= MockPhilloxConverter functions =======
MockPhilloxConverter::MockPhilloxConverter(char* out,
                                           const element::Type& elem_type,
                                           const size_t elem_count,
                                           const char* min_val,
                                           const char* max_val,
                                           const size_t uints_generated_per_exec)
    : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, uints_generated_per_exec) {}

void MockPhilloxConverter::convert(PhilloxOutput result, size_t idx) {
    // Mock converter does nothing
    return;
}

size_t MockPhilloxConverter::get_converted_elements_count() const {
    return 4;
}

// ======= TensorflowPhilloxConverter functions =======

TensorflowPhilloxConverter::TensorflowPhilloxConverter(char* out,
                                                       const element::Type& elem_type,
                                                       const size_t elem_count,
                                                       const char* min_val,
                                                       const char* max_val,
                                                       const size_t uints_generated_per_exec)
    : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, uints_generated_per_exec) {}

size_t TensorflowPhilloxConverter::get_converted_elements_count() const {
    // Each run of Philox algorithm generates 4 uint32 values.
    // If output_type is int32, f32, bf16, or f16 each value is converted to
    // corresponding type so we have 4 result values. For f64 and i64 we use
    // a pair of values for conversion, so we have 2 result values.
    // uints_generated_per_exec indicates how many values we generate in one iteration.
    return m_elem_type.size() > 4 ? 2 : 4;
}

void TensorflowPhilloxConverter::convert(PhilloxOutput result, size_t idx) {
    // convert values to corresponding output_type
    switch (m_elem_type) {
    case element::Type_t::f32: {
        convert_to_output_type<float>(result,
                                      m_uints_generated_per_exec,
                                      get_converted_elements_count(),
                                      m_elem_type,
                                      m_min_val,
                                      m_max_val,
                                      m_out,
                                      idx,
                                      m_elem_count,
                                      uint32_to_float);
        break;
    }
    case element::Type_t::f16: {
        convert_to_output_type<float16>(result,
                                        m_uints_generated_per_exec,
                                        get_converted_elements_count(),
                                        m_elem_type,
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        idx,
                                        m_elem_count,
                                        uint32_to_float16);
        break;
    }
    case element::Type_t::bf16: {
        convert_to_output_type<bfloat16>(result,
                                         m_uints_generated_per_exec,
                                         get_converted_elements_count(),
                                         m_elem_type,
                                         m_min_val,
                                         m_max_val,
                                         m_out,
                                         idx,
                                         m_elem_count,
                                         uint32_to_bfloat16);
        break;
    }
    case element::Type_t::f64: {
        convert_to_output_type<double>(result,
                                       m_uints_generated_per_exec,
                                       get_converted_elements_count(),
                                       m_elem_type,
                                       m_min_val,
                                       m_max_val,
                                       m_out,
                                       idx,
                                       m_elem_count,
                                       nullptr,
                                       [](uint32_t a, uint32_t b, double mn, double mx) {
                                           return uint32_to_double(a, b) * (mx - mn) + mn;
                                       });
        break;
    }
    case element::Type_t::i32: {
        convert_to_output_type<int>(result,
                                    m_uints_generated_per_exec,
                                    get_converted_elements_count(),
                                    m_elem_type,
                                    m_min_val,
                                    m_max_val,
                                    m_out,
                                    idx,
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
                                        m_uints_generated_per_exec,
                                        get_converted_elements_count(),
                                        m_elem_type,
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        idx,
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

// ======= PyTorchPhilloxConverter functions =======

PyTorchPhilloxConverter::PyTorchPhilloxConverter(char* out,
                                                 const element::Type& elem_type,
                                                 const size_t elem_count,
                                                 const char* min_val,
                                                 const char* max_val,
                                                 const size_t uints_generated_per_exec)
    : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, uints_generated_per_exec) {}

size_t PyTorchPhilloxConverter::get_converted_elements_count() const {
    // PyTorch uses one uint32_t value per generated output
    // except for int64 only
    if (m_elem_type.size() > 4) {
        int64_t mn[1];
        int64_t mx[1];
        memcpy(mn, m_min_val, m_elem_type.size());
        memcpy(mx, m_max_val, m_elem_type.size());

        if (mn[0] <= std::numeric_limits<uint32_t>::max() && mx[0] <= std::numeric_limits<uint32_t>::max()) {
            return 4;
        }
        return 2;
    }
    return 4;
    // return m_elem_type.size() > 4 ? 2 : 4;
    // return m_elem_type == element::i64 ? 2 : 4;
}

void PyTorchPhilloxConverter::convert(PhilloxOutput result, size_t idx) {
    // convert values to corresponding output_type
    switch (m_elem_type) {
    case element::Type_t::f32: {
        convert_to_output_type<float>(result,
                                      m_uints_generated_per_exec,
                                      get_converted_elements_count(),
                                      m_elem_type,
                                      m_min_val,
                                      m_max_val,
                                      m_out,
                                      idx,
                                      m_elem_count,
                                      uint32_to_T_type_float<float, uint32_t>);
        break;
    }
    case element::Type_t::f16: {
        convert_to_output_type<float16>(result,
                                        m_uints_generated_per_exec,
                                        get_converted_elements_count(),
                                        m_elem_type,
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        idx,
                                        m_elem_count,
                                        nullptr,
                                        nullptr,
                                        [](uint32_t x, float16 mn, float16 mx) {
                                            auto x_conv = uint32_to_T_type_float<float16, uint32_t>(x);
                                            return static_cast<float16>(x_conv * (mx - mn) + mn);
                                        });
        break;
    }
    case element::Type_t::bf16: {
        convert_to_output_type<bfloat16>(result,
                                         m_uints_generated_per_exec,
                                         get_converted_elements_count(),
                                         m_elem_type,
                                         m_min_val,
                                         m_max_val,
                                         m_out,
                                         idx,
                                         m_elem_count,
                                         nullptr,
                                         nullptr,
                                         [](uint32_t x, bfloat16 mn, bfloat16 mx) {
                                             auto x_conv = uint32_to_bfloat16_pytorch(x);
                                             return bfloat16(x_conv * (mx - mn) + mn);
                                         });
        break;
    }
    case element::Type_t::f64: {
        convert_to_output_type<double>(result,
                                       m_uints_generated_per_exec,
                                       get_converted_elements_count(),
                                       m_elem_type,
                                       m_min_val,
                                       m_max_val,
                                       m_out,
                                       idx,
                                       m_elem_count,
                                       nullptr,
                                       [](uint32_t a, uint32_t b, double mn, double mx) {
                                           uint64_t val = unite_high_low(a, b);
                                           return uint32_to_T_type_float<double, uint64_t>(val) * (mx - mn) + mn;
                                       });
        break;
    }
    case element::Type_t::i32: {
        convert_to_output_type<int>(result,
                                    m_uints_generated_per_exec,
                                    get_converted_elements_count(),
                                    m_elem_type,
                                    m_min_val,
                                    m_max_val,
                                    m_out,
                                    idx,
                                    m_elem_count,
                                    nullptr,
                                    nullptr,
                                    [](uint32_t x, int mn, int mx) {
                                        return static_cast<int>((x) % (mx - mn) + mn);
                                    });
        break;
    }
    case element::Type_t::i64: {
        if (get_converted_elements_count() == 4) {
            convert_to_output_type<int64_t>(result,
                                            m_uints_generated_per_exec,
                                            get_converted_elements_count(),
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            idx,
                                            m_elem_count,
                                            nullptr,
                                            nullptr,
                                            [](uint32_t x, int64_t mn, int64_t mx) {
                                                return (static_cast<int64_t>(x) % (mx - mn)) + mn;
                                            });
        } else {
            convert_to_output_type<int64_t>(result,
                                            m_uints_generated_per_exec,
                                            get_converted_elements_count(),
                                            m_elem_type,
                                            m_min_val,
                                            m_max_val,
                                            m_out,
                                            idx,
                                            m_elem_count,
                                            nullptr,
                                            [](uint32_t a, uint32_t b, int64_t mn, int64_t mx) {
                                                auto val = unite_high_low(a, b);
                                                return static_cast<int64_t>(val % (mx - mn)) + mn;
                                            });
        }
        break;
    }
    default:
        OPENVINO_THROW("Unsupported type of RandomUniform: ", m_elem_type.to_string());
    }
}

// ====== General selector function to construct a desired converter for a generator ======

std::shared_ptr<PhilloxConverter> make_phillox_converter(char* out,
                                                         const element::Type& elem_type,
                                                         const size_t elem_count,
                                                         const char* min_val,
                                                         const char* max_val,
                                                         const std::shared_ptr<PhilloxGenerator> generator) {
    switch (generator->get_alignment()) {
    case PhilloxAlignment::OPENVINO:
    case PhilloxAlignment::TENSORFLOW:
        // Exactly the same conversion for Openvino and Tensorflow
        return std::make_shared<TensorflowPhilloxConverter>(out,
                                                            elem_type,
                                                            elem_count,
                                                            min_val,
                                                            max_val,
                                                            generator->get_generated_elements_count());
    case PhilloxAlignment::PYTORCH:
        // Different conversion
        return std::make_shared<PyTorchPhilloxConverter>(out,
                                                         elem_type,
                                                         elem_count,
                                                         min_val,
                                                         max_val,
                                                         generator->get_generated_elements_count());
    default:
        // Mock conversion (no conversion)
        return std::make_shared<MockPhilloxConverter>(out,
                                                      elem_type,
                                                      elem_count,
                                                      min_val,
                                                      max_val,
                                                      generator->get_generated_elements_count());
    }
}

}  // namespace phillox

}  // namespace reference

}  // namespace ov
