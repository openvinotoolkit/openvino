// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gelu.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

namespace {
OutputVector translate_gelu_common(const NodeContext& context, const std::string& approximate) {
    auto x = context.get_input(0);
    if (approximate == "none") {
        return {context.mark_node(std::make_shared<ov::op::v7::Gelu>(x, ov::op::GeluApproximationMode::ERF))};
    }
    if (approximate == "tanh") {
        return {context.mark_node(std::make_shared<ov::op::v7::Gelu>(x, ov::op::GeluApproximationMode::TANH))};
    }
    PYTORCH_OP_CONVERSION_CHECK(false, "Unsupported approximate for Gelu: ", approximate);
};
}  // namespace

OutputVector translate_gelu(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    std::string approximate = "none";
    if (!context.input_is_none(1)) {
        approximate = context.const_input<std::string>(1);
    }
    return translate_gelu_common(context, approximate);
};

OutputVector translate_gelu_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);
    std::string approximate = "none";
    if (context.has_attribute("approximate")) {
        approximate = context.get_attribute<std::string>("approximate");
    }
    return translate_gelu_common(context, approximate);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
