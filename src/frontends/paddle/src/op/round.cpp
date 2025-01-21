// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs round(const NodeContext& node) {
    return node.default_single_output_mapping(
        {std::make_shared<default_opset::Round>(node.get_input("X"),
                                                ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO)},
        {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
