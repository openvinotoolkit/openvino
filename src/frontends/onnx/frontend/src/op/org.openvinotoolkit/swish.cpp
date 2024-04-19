// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/swish.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/swish.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector swish(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};

    ov::Output<ov::Node> beta;
    if (ov_inputs.size() > 1) {
        beta = ov::frontend::onnx::reshape::interpret_as_scalar(ov_inputs.at(1));
    } else {
        beta = v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0});
    }

    return {std::make_shared<v4::Swish>(ov_inputs.at(0), beta)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
