// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather_elements.hpp"

#include "core/operator_set.hpp"
namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector gather_elements(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ng_inputs{node.get_ov_inputs()};
    auto data = ng_inputs.at(0);
    auto indices = ng_inputs.at(1);
    auto axis = node.get_attribute_value<int64_t>("axis", 0);

    return {std::make_shared<ov::op::v6::GatherElements>(data, indices, axis)};
}
ONNX_OP("GatherElements", OPSET_SINCE(1), ai_onnx::opset_1::gather_elements);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
