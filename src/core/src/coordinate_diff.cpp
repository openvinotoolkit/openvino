// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate_diff.hpp"

#include "openvino/util/common_util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff) {
    s << "CoordinateDiff{";
    s << ov::util::join(coordinate_diff);
    s << "}";
    return s;
}

ov::CoordinateDiff::CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs)
    : std::vector<std::ptrdiff_t>(diffs) {}

ov::CoordinateDiff::CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs) : std::vector<std::ptrdiff_t>(diffs) {}

ov::CoordinateDiff::CoordinateDiff(const CoordinateDiff& diffs) = default;

ov::CoordinateDiff::CoordinateDiff(size_t n, std::ptrdiff_t initial_value)
    : std::vector<std::ptrdiff_t>(n, initial_value) {}

ov::CoordinateDiff::CoordinateDiff() = default;

ov::CoordinateDiff::~CoordinateDiff() = default;

ov::CoordinateDiff& ov::CoordinateDiff::operator=(const CoordinateDiff& v) {
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
    return *this;
}

ov::CoordinateDiff& ov::CoordinateDiff::operator=(CoordinateDiff&& v) noexcept {
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(std::move(v));
    return *this;
}

ov::AttributeAdapter<ov::CoordinateDiff>::~AttributeAdapter() = default;
