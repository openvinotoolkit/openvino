// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reshape.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector reshape(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto data = ov_inputs.at(0);

    ov::Output<ov::Node> pattern;
    bool special_zero = true;
    // Since opset 5 the target shape is provided as input
    if (ov_inputs.size() == 2) {
        pattern = ov_inputs.at(1);
    } else {
        // Added in onnx reshape version 14
        special_zero = !node.get_attribute_value<int64_t>("allowzero", 0);

        pattern = node.get_attribute_as_constant<std::vector<int64_t>>("shape", {});
    }

    return {std::make_shared<v1::Reshape>(data, pattern, special_zero)};
}

ONNX_OP("Reshape", OPSET_SINCE(1), ai_onnx::opset_1::reshape);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
