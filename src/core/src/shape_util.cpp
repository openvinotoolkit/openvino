// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape_util.hpp"

#include <algorithm>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape_util.hpp"

namespace ngraph {
template <>
PartialShape project(const PartialShape& shape, const AxisSet& axes) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (axes.find(i) != axes.end()) {
                result_dims.push_back(shape[i]);
            }
        }

        return PartialShape(result_dims);
    }
}

template <>
PartialShape reduce(const PartialShape& shape, const AxisSet& deleted_axes, bool keep_dims) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        for (int64_t i = 0; i < shape.rank().get_length(); i++) {
            if (deleted_axes.find(i) == deleted_axes.end()) {
                result_dims.push_back(shape[i]);
            } else {
                if (keep_dims)
                    result_dims.emplace_back(1);
            }
        }

        return result_dims;
    }
}

template <>
PartialShape inject_pairs(const PartialShape& shape,
                          std::vector<std::pair<size_t, Dimension>> new_axis_pos_value_pairs) {
    if (shape.rank().is_dynamic()) {
        return shape;
    } else {
        std::vector<Dimension> result_dims;

        size_t original_pos = 0;

        for (size_t result_pos = 0; result_pos < shape.rank().get_length() + new_axis_pos_value_pairs.size();
             result_pos++) {
            auto search_it = std::find_if(new_axis_pos_value_pairs.begin(),
                                          new_axis_pos_value_pairs.end(),
                                          [result_pos](std::pair<size_t, Dimension> p) {
                                              return p.first == result_pos;
                                          });

            if (search_it == new_axis_pos_value_pairs.end()) {
                result_dims.push_back(shape[original_pos++]);
            } else {
                result_dims.push_back(search_it->second);
            }
        }

        return PartialShape{result_dims};
    }
}
}  // namespace ngraph

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
Shape make_dynamic_shape() {
    return Shape{0, std::numeric_limits<size_t>::max()};
}

bool is_dynamic_shape(const Shape& s) {
    OPENVINO_SUPPRESS_DEPRECATED_START
    static const auto dyn_shape = make_dynamic_shape();
    OPENVINO_SUPPRESS_DEPRECATED_END
    return s == dyn_shape;
}

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
}  // namespace util
}  // namespace ov
