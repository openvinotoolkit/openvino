// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/opsets/opset10.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector is_inf(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    ov::opset10::IsInf::Attributes attributes{};
    attributes.detect_negative = node.get_attribute_value<int64_t>("detect_negative", 1);
    attributes.detect_positive = node.get_attribute_value<int64_t>("detect_positive", 1);

    return {std::make_shared<v10::IsInf>(data, attributes)};
}
ONNX_OP("IsInf", OPSET_SINCE(1), ai_onnx::opset_1::is_inf);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
