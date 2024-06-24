// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/elu.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector elu(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 1);

    return {std::make_shared<v0::Elu>(data, alpha)};
}

ONNX_OP("Elu", OPSET_SINCE(1), ai_onnx::opset_1::elu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
