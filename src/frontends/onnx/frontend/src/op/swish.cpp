// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/swish.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/convert.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_24 {

ov::OutputVector swish(const ov::frontend::onnx::Node& node) {
    // Operator definition: https://onnx.ai/onnx/operators/onnx__Swish.html
    //   Y = X * Sigmoid(alpha * X)
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs.at(0);

    // A second input is accepted for backward compatibility with the legacy OpenVINO Swish,
    // which passes beta as an input instead of the standard "alpha" attribute.
    ov::Output<ov::Node> alpha;
    if (inputs.size() > 1) {
        CHECK_VALID_NODE(node,
                         !node.has_attribute("alpha"),
                         "Swish expects either an 'alpha' attribute or a beta input, but not both.");
        alpha = ov::frontend::onnx::reshape::interpret_as_scalar(inputs.at(1));
        if (alpha.get_element_type() != data.get_element_type()) {
            alpha = std::make_shared<v0::Convert>(alpha, data.get_element_type());
        }
    } else {
        alpha = node.get_attribute_as_constant<float>("alpha", 1.0f, data.get_element_type());
    }

    return {std::make_shared<v4::Swish>(data, alpha)};
}

// Swish is a part of the standard ONNX domain since opset 24, but it is registered since opset 1,
// because OperatorsBridge requires a translator for every version imported by a model.
ONNX_OP("Swish", OPSET_SINCE(1), ai_onnx::opset_24::swish);

}  // namespace opset_24
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
