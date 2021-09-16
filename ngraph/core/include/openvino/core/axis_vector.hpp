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
/// \brief A vector of axes.
class AxisVector : public std::vector<size_t> {
public:
    OPENVINO_API AxisVector(const std::initializer_list<size_t>& axes);

    OPENVINO_API AxisVector(const std::vector<size_t>& axes);

    OPENVINO_API AxisVector(const AxisVector& axes);

    OPENVINO_API explicit AxisVector(size_t n);

    template <class InputIterator>
    AxisVector(InputIterator first, InputIterator last) : std::vector<size_t>(first, last) {}

    OPENVINO_API AxisVector();

    OPENVINO_API ~AxisVector();

    OPENVINO_API AxisVector& operator=(const AxisVector& v);

    OPENVINO_API AxisVector& operator=(AxisVector&& v) noexcept;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const AxisVector& axis_vector);

template <>
class OPENVINO_API AttributeAdapter<AxisVector> : public IndirectVectorValueAccessor<AxisVector, std::vector<int64_t>> {
public:
    AttributeAdapter(AxisVector& value) : IndirectVectorValueAccessor<AxisVector, std::vector<int64_t>>(value) {}

    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<AxisVector>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};

}  // namespace ov
