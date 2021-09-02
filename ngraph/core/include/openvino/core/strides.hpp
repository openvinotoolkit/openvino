// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"

namespace ov {
/// \brief Strides for a tensor.
class Strides : public std::vector<size_t> {
public:
    OPENVINO_API Strides();

    OPENVINO_API Strides(const std::initializer_list<size_t>& axis_strides);

    OPENVINO_API Strides(const std::vector<size_t>& axis_strides);

    OPENVINO_API Strides(const Strides& axis_strides);

    OPENVINO_API explicit Strides(size_t n, size_t initial_value = 0);

    template <class InputIterator>
    Strides(InputIterator first, InputIterator last) : std::vector<size_t>(first, last) {}

    OPENVINO_API Strides& operator=(const Strides& v);

    OPENVINO_API Strides& operator=(Strides&& v) noexcept;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const Strides& strides);

template <>
class OPENVINO_API AttributeAdapter<Strides> : public IndirectVectorValueAccessor<Strides, std::vector<int64_t>> {
public:
    AttributeAdapter(Strides& value) : IndirectVectorValueAccessor<Strides, std::vector<int64_t>>(value) {}
    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<Strides>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
