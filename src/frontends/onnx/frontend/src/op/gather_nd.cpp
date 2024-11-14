// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "openvino/op/gather_nd.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector gather_nd(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector ng_inputs{node.get_ov_inputs()};
    const auto data = ng_inputs.at(0);
    const auto indices = ng_inputs.at(1);
    const auto batch_dims = node.get_attribute_value<int64_t>("batch_dims", 0);

    return {std::make_shared<v8::GatherND>(data, indices, batch_dims)};
}

ONNX_OP("GatherND", OPSET_SINCE(1), ai_onnx::opset_1::gather_nd);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
