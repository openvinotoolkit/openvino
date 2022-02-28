// Copyright (C) 2018-2022 Intel Corporation
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
    const auto output_names = node.get_output_var_names("Out");

    auto placehodler = std::make_shared<ov::op::internal::TensorArrayWrite>(x, index, output_names[0]);

    return node.default_single_output_mapping({placehodler}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov