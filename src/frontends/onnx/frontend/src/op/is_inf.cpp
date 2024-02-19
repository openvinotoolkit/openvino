// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/is_inf.hpp"

#include "openvino/opsets/opset10.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector is_inf(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    ov::opset10::IsInf::Attributes attributes{};
    attributes.detect_negative = node.get_attribute_value<int64_t>("detect_negative", 1);
    attributes.detect_positive = node.get_attribute_value<int64_t>("detect_positive", 1);

    return {std::make_shared<v10::IsInf>(data, attributes)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
