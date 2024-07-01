// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/strides.hpp"

#include "openvino/util/common_util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const ov::Strides& strides) {
    s << "Strides{";
    s << ov::util::join(strides);
    s << "}";
    return s;
}

ov::Strides::Strides() : std::vector<size_t>() {}

ov::Strides::Strides(const std::initializer_list<size_t>& axis_strides) : std::vector<size_t>(axis_strides) {}

ov::Strides::Strides(const std::vector<size_t>& axis_strides) : std::vector<size_t>(axis_strides) {}

ov::Strides::Strides(const Strides& axis_strides) : std::vector<size_t>(axis_strides) {}

ov::Strides::Strides(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ov::Strides& ov::Strides::operator=(const Strides& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::Strides& ov::Strides::operator=(Strides&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}
