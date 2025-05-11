// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/shape_util.hpp"

#include <algorithm>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

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
}  // namespace util
}  // namespace ov
