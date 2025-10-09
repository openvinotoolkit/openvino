// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/convert.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector identity(const ov::frontend::onnx::Node& node) {
    // Input
    ov::Output<ov::Node> input = node.get_ov_inputs().at(0);
    return {std::make_shared<v0::Convert>(input, input.get_element_type())};
}
ONNX_OP("Identity", OPSET_SINCE(1), ai_onnx::opset_1::identity);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
