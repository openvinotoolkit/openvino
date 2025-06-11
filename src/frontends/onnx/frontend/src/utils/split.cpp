// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/split.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ov {
namespace op {
namespace util {
OutputVector make_split(const Output<ov::Node>& value, const std::vector<int64_t>& split_lengths, int64_t axis) {
    const auto axis_node = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {axis});
    const auto split_lengths_node =
        ov::op::v0::Constant::create(ov::element::i64, Shape{split_lengths.size()}, split_lengths);
    const auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(value, axis_node, split_lengths_node);

    return variadic_split->outputs();
}

OutputVector make_split(const Output<ov::Node>& value, int64_t num_splits, int64_t axis) {
    auto value_shape = value.get_partial_shape();
    if (value_shape.rank().is_static() && value_shape.size() > static_cast<size_t>(axis) &&
        value_shape[axis].is_static()) {
        auto axis_len = value_shape[axis].get_length();
        // In ONNX definition, if the tensor is not evenly splittable the last chunk will be smaller.
        // Handle the case when axis_len is not divisible by num_splits
        if (axis_len % num_splits) {
            auto avg_axis = axis_len / num_splits + 1;
            auto last_output_value = axis_len % avg_axis;
            std::vector<int64_t> split_lengths(num_splits, avg_axis);
            split_lengths.back() = last_output_value;
            return make_split(value, split_lengths, axis);
        }
    }
    const auto axis_node = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {axis});
    const auto split = std::make_shared<ov::op::v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}
}  // namespace util
}  // namespace op
}  // namespace ov
