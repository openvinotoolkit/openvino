// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/swish.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
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

ONNX_OP("Swish", OPSET_SINCE(1), org_openvinotoolkit::opset_1::swish, OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
