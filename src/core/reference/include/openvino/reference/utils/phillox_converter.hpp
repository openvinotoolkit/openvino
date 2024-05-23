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

class PhilloxConverter {
public:
    PhilloxConverter() = delete;

    virtual ~PhilloxConverter(){};

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given
    /// dtype
    virtual size_t get_converted_elements_count() const = 0;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the
    /// output array.
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

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given
    /// dtype
    size_t get_converted_elements_count() const override;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the
    /// output array.
    void convert(PhilloxOutput result, size_t idx) override;
};

class PyTorchPhilloxConverter : public PhilloxConverter {
public:
    PyTorchPhilloxConverter() = delete;

    PyTorchPhilloxConverter(char* out,
                            const element::Type& elem_type,
                            const size_t elem_count,
                            const char* min_val,
                            const char* max_val,
                            const size_t step)
        : PhilloxConverter(out, elem_type, elem_count, min_val, max_val, step) {}

    /// \brief Returns the number of generated elements based on the number of generator output elements and the given
    /// dtype
    size_t get_converted_elements_count() const override;

    /// \brief Converts the given array (PhilloxOutput) to the target dtype and assigns them at the k-th index of the
    /// output array.
    void convert(PhilloxOutput result, size_t idx) override;
};

/// \brief Constructs and returns a shared pointer to the converter matching to the provided gtenerator.
std::shared_ptr<PhilloxConverter> make_phillox_converter(char* out,
                                                         const element::Type& elem_type,
                                                         const size_t elem_count,
                                                         const char* min_val,
                                                         const char* max_val,
                                                         const std::shared_ptr<PhilloxGenerator> generator);

}  // namespace phillox
}  // namespace reference
}  // namespace ov
