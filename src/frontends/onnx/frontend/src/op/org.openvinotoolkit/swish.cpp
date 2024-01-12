// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/swish.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/swish.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector swish(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};

    Output<ov::Node> beta;
    if (ng_inputs.size() > 1) {
        beta = ngraph::onnx_import::reshape::interpret_as_scalar(ng_inputs.at(1));
    } else {
        beta = v0::Constant::create(element::f32, Shape{}, {1.0});
    }

    return {std::make_shared<v4::Swish>(ng_inputs.at(0), beta)};
}

}  // namespace set_1
}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
