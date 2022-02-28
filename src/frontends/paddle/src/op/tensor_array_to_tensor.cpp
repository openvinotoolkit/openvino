// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <limits.h>

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs tensor_array_to_tensor(const NodeContext& node) {
    using namespace default_opset;
    const auto x = node.get_input("X");
    auto axis = node.get_attribute<int32_t>("axis", 0);
    PADDLE_OP_CHECK(node, axis == 0, "axis should be 0, got: ", axis);

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