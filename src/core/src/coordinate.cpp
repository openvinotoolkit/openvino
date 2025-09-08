// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/coordinate.hpp"

#include "openvino/util/common_util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const Coordinate& coordinate) {
    s << "Coordinate{";
    s << ov::util::join(coordinate);
    s << "}";
    return s;
}

ov::Coordinate::Coordinate() = default;

ov::Coordinate::Coordinate(const std::initializer_list<size_t>& axes) : std::vector<size_t>(axes) {}

ov::Coordinate::Coordinate(const ov::Shape& shape)
    : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape)) {}

ov::Coordinate::Coordinate(const std::vector<size_t>& axes) : std::vector<size_t>(axes) {}

ov::Coordinate::Coordinate(const Coordinate& axes) = default;

ov::Coordinate::Coordinate(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ov::Coordinate::~Coordinate() = default;

ov::Coordinate& ov::Coordinate::operator=(const Coordinate& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::Coordinate& ov::Coordinate::operator=(Coordinate&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

ov::AttributeAdapter<ov::Coordinate>::~AttributeAdapter() = default;
