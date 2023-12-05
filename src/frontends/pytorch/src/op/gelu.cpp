// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_gelu(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    std::string approximate = "none";
    if (!context.input_is_none(1)) {
        approximate = context.const_input<std::string>(1);
    }
    if (approximate == "none") {
        return {context.mark_node(std::make_shared<ov::op::v7::Gelu>(x, ov::op::GeluApproximationMode::ERF))};
    }
    if (approximate == "tanh") {
        return {context.mark_node(std::make_shared<ov::op::v7::Gelu>(x, ov::op::GeluApproximationMode::TANH))};
    }
    FRONT_END_OP_CONVERSION_CHECK(false, "Unsupported approximate for Gelu: ", approximate);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
