// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/philox_converter.hpp"

namespace ov {

namespace reference {

namespace philox {

// Utils functions
namespace {

/// ================ PYTORCH ======================

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

// Helper function for converting uint values to any T type float, using PyTorch
// conversion method. Sets fractional part of floating value with bits from
// uint32 value. Resulting value is in interval [0,1).
template <typename T, typename V>
PytorchAccumulatorType<T> pytorch_uint_to_float(V x) {
    const auto mask = static_cast<V>((uint64_t(1) << std::numeric_limits<T>::digits) - 1);
    const auto divisor = static_cast<PytorchAccumulatorType<T>>(1) / (uint64_t(1) << std::numeric_limits<T>::digits);
    PytorchAccumulatorType<T> ret = (x & mask) * divisor;
    return ret;
}

float pytorch_uint_to_bfloat(uint32_t x) {
    const auto mask = static_cast<uint32_t>((1UL << 8) - 1);
    const auto divisor = static_cast<PytorchAccumulatorType<ov::bfloat16>>(1) / (1UL << 8);
    PytorchAccumulatorType<ov::bfloat16> ret = (x & mask) * divisor;
    return ret;
}

/// ================ TENSORFLOW ======================

// Helper struct for converting between types in Tensorflow
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

    auto x_uint16 = static_cast<uint16_t>(x);
    convert_types out_val = {(static_cast<uint16_t>(127) << 7) | (x_uint16 & 0x7fu)};
    return out_val.bf16 - static_cast<bfloat16>(1);
}

/// \brief General util function used to convert uint32 values to destination type and normalize to required range.
/// \param generated_numbers The set of uint32s generated from a generator
/// \param converted_output_values_per_exec_count The amount of values a converter outputs when given the set of
/// generated numbers \param min_val The pointer to the minimum value \param max_val The pointer to the maximum value
/// \param out The pointer to the output location to place the converted values
/// \param idx The current starting index to place the values at in the 'out' array
/// \param elem_count The total number of elements (size of 'out' array)
/// \param convert_single_input A lambda to convert one uint32 to a given dtype (auto normalization)
/// \param convert_two_inputs A lambda to convert two uint32s to a given dtype (normalization required)
/// \param mod_func A lambda to convert one uint32 to a given dtype (normalization required)
template <typename T>
void convert_to_output_type(const std::vector<uint32_t>& generated_numbers,
                            size_t converted_output_values_per_exec_count,
                            const char* min_val,
                            const char* max_val,
                            char* out,
                            size_t idx,
                            size_t elem_count,
                            T (*convert_single_input)(uint32_t) = nullptr,
                            T (*convert_two_inputs)(uint32_t, uint32_t, T, T) = nullptr,
                            T (*mod_func)(uint32_t, T, T) = nullptr) {
    // Get min and max values
    const auto mn = *reinterpret_cast<const T*>(min_val);
    const auto mx = *reinterpret_cast<const T*>(max_val);

    std::vector<T> output_elements_buffer(converted_output_values_per_exec_count);

    if (converted_output_values_per_exec_count == ELEMENTS_PER_EXECUTION) {
        // Each element of resulting sequence is formed using single uint32 value,
        // therefore the number of output elements is equal to number of generated uint32s.
        std::transform(generated_numbers.data(),
                       generated_numbers.data() + ELEMENTS_PER_EXECUTION,
                       output_elements_buffer.data(),
                       [&mn, &mx, &convert_single_input, &mod_func](uint32_t elem) {
                           if (convert_single_input != nullptr) {
                               return convert_single_input(elem) * (mx - mn) + mn;
                           } else {
                               return mod_func(elem, mn, mx);
                           }
                       });
    } else if (converted_output_values_per_exec_count == ELEMENTS_PER_EXECUTION / 2) {
        // Each element of resulting sequence is formed using two uint32 values,
        // therefore the number of output elements is half of ELEMENTS_PER_EXECUTION.
        for (size_t i = 0; i < converted_output_values_per_exec_count; ++i) {
            output_elements_buffer[i] =
                convert_two_inputs(generated_numbers[2 * i], generated_numbers[2 * i + 1], mn, mx);
        }
    } else {
        // This behavior might change one day, if 128 bit numbers are possible
        OPENVINO_THROW("The converter has requested an incorrect number of output values: ",
                       converted_output_values_per_exec_count,
                       " (possible ",
                       ELEMENTS_PER_EXECUTION / 2,
                       " or ",
                       ELEMENTS_PER_EXECUTION,
                       ")");
    }

    memcpy(out + idx * sizeof(T),
           output_elements_buffer.data(),
           std::min(converted_output_values_per_exec_count, elem_count - idx) * sizeof(T));
}
}  // namespace

// ======= MockPhiloxConverter functions =======
MockPhiloxConverter::MockPhiloxConverter(char* out,
                                         const element::Type& elem_type,
                                         const size_t elem_count,
                                         const char* min_val,
                                         const char* max_val)
    : PhiloxConverter(out, elem_type, elem_count, min_val, max_val) {}

void MockPhiloxConverter::convert(PhiloxOutput result, size_t idx) {
    // Mock converter does nothing
    return;
}

size_t MockPhiloxConverter::get_converted_elements_count() const {
    return ELEMENTS_PER_EXECUTION;
}

// ======= TensorflowPhiloxConverter functions =======

TensorflowPhiloxConverter::TensorflowPhiloxConverter(char* out,
                                                     const element::Type& elem_type,
                                                     const size_t elem_count,
                                                     const char* min_val,
                                                     const char* max_val)
    : PhiloxConverter(out, elem_type, elem_count, min_val, max_val) {}

size_t TensorflowPhiloxConverter::get_converted_elements_count() const {
    // Each run of Philox algorithm generates 4 uint32 values.
    // If output_type is int32, f32, bf16, or f16 each value is converted to
    // corresponding type so we have 4 result values.
    // For f64 and i64 we use a pair of values for conversion (2 values),
    // so we have 2 result values.
    return m_elem_type.size() > 4 ? ELEMENTS_PER_EXECUTION / 2 : ELEMENTS_PER_EXECUTION;
}

void TensorflowPhiloxConverter::convert(const PhiloxOutput result, size_t idx) {
    // convert values to corresponding output_type
    switch (m_elem_type) {
    case element::Type_t::f32: {
        convert_to_output_type<float>(result,
                                      get_converted_elements_count(),
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
                                        get_converted_elements_count(),
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
                                         get_converted_elements_count(),
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
                                       get_converted_elements_count(),
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
                                    get_converted_elements_count(),
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
                                        get_converted_elements_count(),
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

// ======= PyTorchPhiloxConverter functions =======

PyTorchPhiloxConverter::PyTorchPhiloxConverter(char* out,
                                               const element::Type& elem_type,
                                               const size_t elem_count,
                                               const char* min_val,
                                               const char* max_val)
    : PhiloxConverter(out, elem_type, elem_count, min_val, max_val) {
    // Check for optimization conditions for int64_t.
    // If both min and max fall below the maximum value of uint32_t,
    // PyTorch generates 64-bit numbers by casting
    // a single 32 bit random number to 64 bit,
    // instead of using 2 32 bit numbers.
    if (elem_type == element::i64) {
        int64_t mn, mx;
        memcpy(&mn, min_val, elem_type.size());
        memcpy(&mx, max_val, elem_type.size());
        m_optimization_enabled =
            mn <= std::numeric_limits<uint32_t>::max() && mx <= std::numeric_limits<uint32_t>::max();
    } else {
        m_optimization_enabled = false;
    }
}

size_t PyTorchPhiloxConverter::get_converted_elements_count() const {
    // PyTorch uses one uint32_t value per generated output for 32 bit random numbers
    // and either one or two values (if optimization is off) to generate one 64 bit random number.
    // Therefore, the only case where 2 output values are generated is when optimization is OFF and the dtype is 64 bit
    // (size > 4)
    return m_elem_type.size() > 4 && !m_optimization_enabled ? ELEMENTS_PER_EXECUTION / 2 : ELEMENTS_PER_EXECUTION;
}

void PyTorchPhiloxConverter::convert(const PhiloxOutput result, size_t idx) {
    // convert values to corresponding output_type
    switch (m_elem_type) {
    case element::Type_t::f32: {
        convert_to_output_type<float>(result,
                                      get_converted_elements_count(),
                                      m_min_val,
                                      m_max_val,
                                      m_out,
                                      idx,
                                      m_elem_count,
                                      pytorch_uint_to_float<float, uint32_t>);
        break;
    }
    case element::Type_t::f16: {
        convert_to_output_type<float16>(result,
                                        get_converted_elements_count(),
                                        m_min_val,
                                        m_max_val,
                                        m_out,
                                        idx,
                                        m_elem_count,
                                        nullptr,
                                        nullptr,
                                        [](uint32_t x, float16 mn, float16 mx) {
                                            auto x_conv = pytorch_uint_to_float<float16, uint32_t>(x);
                                            return static_cast<float16>(x_conv * (mx - mn) + mn);
                                        });
        break;
    }
    case element::Type_t::bf16: {
        convert_to_output_type<bfloat16>(result,
                                         get_converted_elements_count(),
                                         m_min_val,
                                         m_max_val,
                                         m_out,
                                         idx,
                                         m_elem_count,
                                         nullptr,
                                         nullptr,
                                         [](uint32_t x, bfloat16 mn, bfloat16 mx) {
                                             auto x_conv = pytorch_uint_to_bfloat(x);
                                             return bfloat16(x_conv * (mx - mn) + mn);
                                         });
        break;
    }
    case element::Type_t::f64: {
        convert_to_output_type<double>(result,
                                       get_converted_elements_count(),
                                       m_min_val,
                                       m_max_val,
                                       m_out,
                                       idx,
                                       m_elem_count,
                                       nullptr,
                                       [](uint32_t a, uint32_t b, double mn, double mx) {
                                           uint64_t val = unite_high_low(a, b);
                                           return pytorch_uint_to_float<double, uint64_t>(val) * (mx - mn) + mn;
                                       });
        break;
    }
    case element::Type_t::i32: {
        convert_to_output_type<int>(result,
                                    get_converted_elements_count(),
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
                                            get_converted_elements_count(),
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
                                            get_converted_elements_count(),
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

std::shared_ptr<PhiloxConverter> make_philox_converter(char* out,
                                                       const element::Type& elem_type,
                                                       const size_t elem_count,
                                                       const char* min_val,
                                                       const char* max_val,
                                                       const op::PhiloxAlignment alignment) {
    switch (alignment) {
    case op::PhiloxAlignment::TENSORFLOW:
        return std::make_shared<TensorflowPhiloxConverter>(out, elem_type, elem_count, min_val, max_val);
    case op::PhiloxAlignment::PYTORCH:
        return std::make_shared<PyTorchPhiloxConverter>(out, elem_type, elem_count, min_val, max_val);
    default:
        // Mock conversion (no conversion)
        return std::make_shared<MockPhiloxConverter>(out, elem_type, elem_count, min_val, max_val);
    }
}

}  // namespace philox

}  // namespace reference

}  // namespace ov
