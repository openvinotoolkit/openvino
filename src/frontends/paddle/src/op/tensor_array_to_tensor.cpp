// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
// Paddle TensorArray is not natively supported by OpenVINO.
// Here paddle frontend only partially support following circumstances:
// 1. TensorArray could be indexed with paddle slice op.
// We only support slice along axis 0 with only 1 element in TensorArray for now,
// with unsqueezing the element along axis 0 manually.
// 2. TensoArray could be tranfered to tensor with paddle tensor_array_to_tensor op.
// The elements in it should be concated along an axis.
// We only support concat along axis 0 for now. paddle.concat always along axis 0.
// what's more, we only support the pattern of "TensorArrayLength<->TensorArrayWrite" for now, which
// is tranformed togther. That means, tensorarray are always appended at the end.
NamedOutputs tensor_array_to_tensor(const NodeContext& node) {
    using namespace default_opset;
    const auto x = node.get_input("X");
    auto axis = node.get_attribute<int32_t>("axis", 0);
    PADDLE_OP_CHECK(node, axis == 0, "axis should be 0, got: ", axis);

    // All elements in TensorArray have already been unsqueezed and concated. So here
    // just squeeze. We use reshape instead because the tensorarray could be empty, and
    // it looks squeeze would fail with empty tensor.
    const auto shape = std::make_shared<ShapeOf>(x, element::Type_t::i32);
    const auto const_1_node = Constant::create(element::i32, {1}, {1});
    const auto const_max_node = Constant::create(element::i32, {1}, {INT_MAX});
    const auto new_shape = std::make_shared<StridedSlice>(shape,
                                                          const_1_node,
                                                          const_max_node,
                                                          std::vector<int64_t>{0},
                                                          std::vector<int64_t>{0});

    auto placeholder = std::make_shared<Reshape>(x, new_shape, false);

    return node.default_single_output_mapping({placeholder}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
