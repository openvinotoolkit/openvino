// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/rtti.hpp"

namespace ov {
/// \brief Strides for a tensor.
class OPENVINO_API Strides : public std::vector<size_t> {
public:
    Strides();

    Strides(const std::initializer_list<size_t>& axis_strides);

    Strides(const std::vector<size_t>& axis_strides);

    Strides(const Strides& axis_strides);

    explicit Strides(size_t n, size_t initial_value = 0);

    template <class InputIterator>
    Strides(InputIterator first, InputIterator last) : std::vector<size_t>(first, last) {}

    Strides& operator=(const Strides& v);

    Strides& operator=(Strides&& v) noexcept;
};

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const Strides& strides);

template <>
class OPENVINO_API AttributeAdapter<Strides> : public IndirectVectorValueAccessor<Strides, std::vector<int64_t>> {
public:
    AttributeAdapter(Strides& value) : IndirectVectorValueAccessor<Strides, std::vector<int64_t>>(value) {}
    OPENVINO_RTTI("AttributeAdapter<Strides>");
    BWDCMP_RTTI_DECLARATION;
};

}  // namespace ov
