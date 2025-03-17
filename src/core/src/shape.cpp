// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape.hpp"

#include "openvino/core/shape_util.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/common_util.hpp"

std::ostream& ov::operator<<(std::ostream& s, const Shape& shape) {
    s << "[";
    s << ov::util::join(shape, ",");
    s << "]";
    return s;
}

namespace {
size_t stringToSizeT(const std::string& valStr) {
    size_t ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}
}  // namespace

ov::Shape::Shape() : std::vector<size_t>() {}

ov::Shape::Shape(const std::initializer_list<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::Shape::Shape(const std::vector<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::Shape::Shape(const Shape& axis_lengths) = default;

ov::Shape::Shape(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ov::Shape::Shape(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val[0] == '[' && val[val.size() - 1] == ']')
        val = val.substr(1, val.size() - 2);
    val = ov::util::trim(val);
    std::vector<size_t> dims;
    std::stringstream ss(val);
    std::string field;
    while (getline(ss, field, ',')) {
        OPENVINO_ASSERT(!field.empty(), "Cannot get vector of dimensions! \"" + value + "\" is incorrect");
        dims.insert(dims.end(), stringToSizeT(field));
    }
    *this = dims;
}

ov::Shape::~Shape() = default;

ov::Shape& ov::Shape::operator=(const Shape& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::Shape& ov::Shape::operator=(Shape&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

std::string ov::Shape::to_string() const {
    std::stringstream shape_str_stream;
    shape_str_stream << *this;
    return shape_str_stream.str();
}

namespace ov {

typename Shape::reference Shape::operator[](std::ptrdiff_t i) {
    return std::vector<size_t>::operator[](util::normalize(i, size()));
}

typename Shape::const_reference Shape::operator[](std::ptrdiff_t i) const {
    return std::vector<size_t>::operator[](util::normalize(i, size()));
}

typename Shape::reference Shape::at(std::ptrdiff_t i) {
    return std::vector<size_t>::operator[](util::normalize_shape_index(i, size()));
}

typename Shape::const_reference Shape::at(std::ptrdiff_t i) const {
    return std::vector<size_t>::operator[](util::normalize_shape_index(i, size()));
}

AttributeAdapter<ov::Shape>::~AttributeAdapter() = default;
}  // namespace ov
