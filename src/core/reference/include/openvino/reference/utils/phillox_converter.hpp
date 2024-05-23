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

namespace ov {

namespace reference {

namespace phillox {

// Utils functions from OpenVINO, Tensorflow, PyTorch
namespace {

// Concatenates two uint32 values into single uint64 values.
uint64_t unite_high_low(uint32_t high, uint32_t low);

// Helper function for converting uint32 values to float32. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float uint32_to_float(uint32_t x);

// Helper function for converting uint32 values to float16.Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
float16 uint32_to_float16(uint32_t x);

// Helper function for converting uint32 values to double. Sets fractional part of
// floating double with bits from uint32 values. Resulting value is in interval [0,1).
double uint32_to_double(uint32_t x1, uint32_t x2);

// Helper function for converting uint32 values to bfloat16. Sets fractional part of
// floating value with bits from uint32 value. Resulting value is in interval [0,1).
bfloat16 uint32_to_bfloat16(uint32_t x);

// Helper function for converting uint32 values to any T type float, using PyTorch 
// conversion method. Sets fractional part of floating value with bits from 
// uint32 value. Resulting value is in interval [0,1).
template <typename T>
T uint32_to_T_type_float(uint32_t x);

// Converts uint32 values to destination type and normalizes to required range.
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
                            T (*mod_func)(uint32_t, T, T) = nullptr);

}  // namespace

class PhilloxConverter {
public:
    PhilloxConverter() = delete;

    virtual ~PhilloxConverter() {};

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given dtype
    virtual size_t get_converted_elements_count() const = 0;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the output array.
    virtual void convert(PhilloxOutput result, size_t k) = 0;

protected:
    PhilloxConverter(char* out,
                     const element::Type& elem_type,
                     const size_t elem_count,
                     const char* min_val,
                     const char* max_val,
                     const size_t step)
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
                               const element::Type& elem_type,
                               const size_t elem_count,
                               const char* min_val,
                               const char* max_val,
                               const size_t step)
        : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, step) {}

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given dtype
    size_t get_converted_elements_count() const override;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the output array.
    void convert(PhilloxOutput result, size_t idx) override;
};

class PyTorchPhilloxConverter : public PhilloxConverter {
public:
    PyTorchPhilloxConverter() = delete;

    PyTorchPhilloxConverter(char* out,
                            const size_t step,
                            const element::Type& elem_type,
                            const size_t elem_count,
                            const char* min_val,
                            const char* max_val)
        : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, step) {}

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given dtype
    size_t get_converted_elements_count() const override;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the output array.
    void convert(PhilloxOutput result, size_t idx) override;
};

/// \brief Constructs and returns a shared pointer to the converter matching to the provided gtenerator.
static std::shared_ptr<PhilloxConverter> make_phillox_converter(char* out,
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
        OPENVINO_THROW("Unknown Phillox algorithm alignment option selected.");
    }
}

}  // namespace phillox
}  // namespace reference
}  // namespace ov
