// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/multiply.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector thresholded_relu(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    const double alpha = node.get_attribute_value<double>("alpha", 1.0);

    const auto alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{}, {alpha});

    const auto data_map =
        std::make_shared<v0::Convert>(std::make_shared<v1::Greater>(data, alpha_node), data.get_element_type());

    return {std::make_shared<v1::Multiply>(data, data_map)};
}

ONNX_OP("ThresholdedRelu", OPSET_SINCE(1), ai_onnx::opset_1::thresholded_relu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
