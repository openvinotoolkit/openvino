// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs softplus(const NodeContext& node) {
    auto data = node.get_input("X");
    auto beta = node.get_attribute<float>("beta");
    auto threshold = node.get_attribute<float>("threshold");
    float supported_beta = 1.0;
    float supported_threshold = 20.0;
    const float EPSINON = 1e-6f;

    // Fast path: the only (beta, threshold) pair OpenVINO SoftPlus can express directly.
    if (std::fabs(beta - supported_beta) <= EPSINON && std::fabs(threshold - supported_threshold) <= EPSINON) {
        return node.default_single_output_mapping({std::make_shared<default_opset::SoftPlus>(data)}, {"Out"});
    }

    // General form (Paddle):
    //   softplus(x) = x                          if beta*x >  threshold
    //                (1/beta)*log(1+exp(beta*x)) otherwise
    const auto et = data.get_element_type();
    PADDLE_OP_CHECK(node, et.is_real(), "softplus: X must have a floating-point data type");

    const auto beta_c = std::make_shared<default_opset::Constant>(et, Shape{}, beta);
    const auto inv_beta_c = std::make_shared<default_opset::Constant>(et, Shape{}, 1.0f / beta);
    const auto threshold_c = std::make_shared<default_opset::Constant>(et, Shape{}, threshold);

    auto scaled = std::make_shared<default_opset::Multiply>(data, beta_c);
    // Clamp the exp() argument at `threshold` for numerical stability; the saturated
    // elements are replaced by `x` below and never consume this value.
    auto clamped = std::make_shared<default_opset::Minimum>(scaled, threshold_c);
    auto sp = std::make_shared<default_opset::SoftPlus>(clamped);
    auto out = std::make_shared<default_opset::Multiply>(sp, inv_beta_c);
    auto saturated = std::make_shared<default_opset::Greater>(scaled, threshold_c);
    auto result = std::make_shared<default_opset::Select>(saturated, data, out);
    return node.default_single_output_mapping({result}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
