// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "internal/op/tensorarray_write.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs write_to_array(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto index = node.get_input("I");

    auto placehodler = std::make_shared<ov::op::internal::TensorArrayWrite>(x, index);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
