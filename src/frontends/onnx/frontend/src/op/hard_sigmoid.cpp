// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/hard_sigmoid.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/hard_sigmoid.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector hard_sigmoid(const Node& node) {
    const auto data = node.get_ng_inputs().at(0);

    const auto alpha =
        v0::Constant::create<double>(data.get_element_type(),
                                     Shape{},
                                     std::vector<double>{node.get_attribute_value<double>("alpha", 0.2)});

    const auto beta = v0::Constant::create<double>(data.get_element_type(),
                                                   Shape{},
                                                   std::vector<double>{node.get_attribute_value<double>("beta", 0.5)});

    return {std::make_shared<v0::HardSigmoid>(data, alpha, beta)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
