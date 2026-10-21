// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape_util.hpp"

#include <algorithm>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/math_util.hpp"

namespace ov {
template <class TContainer, class TAxes>
TContainer reduce_container(const TContainer& input, const TAxes& axes) {
    TContainer result;
    const auto input_size = input.size();
    result.reserve(input_size);

    for (size_t axis = 0; axis < input_size; ++axis) {
        if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
            result.emplace_back(input[axis]);
        }
    }

    return result;
}

template <class TContainer, class TAxes>
TContainer replace_container(const TContainer& input, const TAxes& axes) {
    auto result = input;
    for (auto&& axis : axes) {
        result[axis] = 1;
    }
    return result;
}

namespace util {

constexpr size_t bits_per_byte = 8;

Shape reduce(const Shape& input, const AxisSet& axes) {
    return ov::reduce_container(input, axes);
}

Shape reduce(const Shape& input, const AxisSet& axes, const bool keep_dims) {
    return keep_dims ? reduce_keep_dims(input, axes) : reduce(input, axes);
}

std::vector<size_t> reduce(const std::vector<size_t>& input, const AxisSet& axes) {
    return ov::reduce_container(input, axes);
}

Shape reduce_keep_dims(const Shape& input, const AxisSet& axes) {
    return ov::replace_container(input, axes);
}

Shape get_broadcast_shape(const Shape& first, const Shape& second, const op::AutoBroadcastSpec& broadcast_spec) {
    auto out_shape = PartialShape(first);
    OPENVINO_ASSERT(PartialShape::broadcast_merge_into(out_shape, second, broadcast_spec),
                    "Argument shapes are inconsistent");
    return out_shape.to_shape();
}

std::ptrdiff_t normalize_shape_index(std::ptrdiff_t idx, size_t rank) {
    idx = normalize(idx, static_cast<int64_t>(rank));
    if (static_cast<decltype(rank)>(idx) >= rank) {
        OPENVINO_THROW("Accessing out-of-range dimension");
    } else {
        return idx;
    }
}

std::optional<size_t> shape_size_safe(const Shape& shape) {
    size_t size = 1;
    for (auto first = shape.cbegin(), last = shape.cend(); first != last; ++first) {
        if (mul_overflow(size, *first, size)) {
            return std::nullopt;
        }
    }
    return std::make_optional(size);
}

bool has_byte_strides(const element::Type& type, const Shape& shape) {
    const auto bitwidth = type.bitwidth();
    return bitwidth >= bits_per_byte || (bitwidth > 0 && (bits_per_byte % bitwidth) == 0 && !shape.empty() &&
                                         (shape.back() * bitwidth) % bits_per_byte == 0);
}

Strides get_dense_byte_strides(const element::Type& type, const Shape& shape) {
    Strides strides(shape.size());
    if (!strides.empty()) {
        const auto bitwidth = type.bitwidth();
        strides.back() = shape.back() == 0 ? 0 : type.size();

        if (strides.size() > 1) {
            const auto row_byte_size =
                bitwidth < bits_per_byte ? shape.back() * bitwidth / bits_per_byte : shape.back() * type.size();
            strides[strides.size() - 2] = row_byte_size;
            for (auto i = strides.size() - 2; i > 0; --i) {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
    }
    return strides;
}

Shape get_shape_for_byte_strides(const element::Type& type, const Shape& shape) {
    auto byte_shape = shape;
    const auto bitwidth = type.bitwidth();
    if (bitwidth < bits_per_byte && !byte_shape.empty()) {
        byte_shape.back() = byte_shape.back() * bitwidth / bits_per_byte;
    }
    return byte_shape;
}

}  // namespace util
}  // namespace ov
