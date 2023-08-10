// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/is_inf.hpp"

#include "openvino/opsets/opset10.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector is_inf(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    ov::opset10::IsInf::Attributes attributes{};
    attributes.detect_negative = node.get_attribute_value<int64_t>("detect_negative", 1);
    attributes.detect_positive = node.get_attribute_value<int64_t>("detect_positive", 1);

    return {std::make_shared<ov::opset10::IsInf>(data, attributes)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
