// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/axis_vector.hpp"

#include "openvino/util/common_util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const AxisVector& axis_vector) {
    s << "AxisVector{";
    s << ov::util::join(axis_vector);
    s << "}";
    return s;
}

ov::AxisVector::AxisVector(const std::initializer_list<size_t>& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(const std::vector<size_t>& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(const AxisVector& axes) : std::vector<size_t>(axes) {}

ov::AxisVector::AxisVector(size_t n) : std::vector<size_t>(n) {}

ov::AxisVector::AxisVector() {}

ov::AxisVector::~AxisVector() {}

ov::AxisVector& ov::AxisVector::operator=(const AxisVector& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::AxisVector& ov::AxisVector::operator=(AxisVector&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}
