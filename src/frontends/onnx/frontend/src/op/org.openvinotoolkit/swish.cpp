// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/swish.hpp"

#include "default_opset.hpp"
#include "op/org.openvinotoolkit/normalize.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector swish(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};

    Output<ov::Node> beta;
    if (ng_inputs.size() > 1) {
        beta = reshape::interpret_as_scalar(ng_inputs.at(1));
    } else {
        beta = default_opset::Constant::create(element::f32, Shape{}, {1.0});
    }

    return {std::make_shared<default_opset::Swish>(ng_inputs.at(0), beta)};
}

}  // namespace set_1
}  // namespace op

}  // namespace onnx_import

}  // namespace ov
