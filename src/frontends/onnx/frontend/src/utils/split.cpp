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
    const auto axis_node = ov::op::v0::Constant::create(ov::element::i64, Shape{}, {axis});
    const auto split = std::make_shared<ov::op::v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}
}  // namespace util
}  // namespace op
}  // namespace ov
