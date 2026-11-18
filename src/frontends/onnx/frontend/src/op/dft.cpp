// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/dft.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"
namespace ov::frontend::onnx::ai_onnx {
namespace {
ov::OutputVector translate_dft(const ov::frontend::onnx::Node& node, const ov::Output<ov::Node>& axes) {
    const auto inputs = node.get_ov_inputs();
    const auto length_provided = inputs.size() > 1 && !ov::op::util::is_null(inputs[1]);
    const auto inverse = node.get_attribute_value<int64_t>("inverse", 0);
    const auto onesided = node.get_attribute_value<int64_t>("onesided", 0);
    return {dft::make_dft(inputs.at(0),
                          length_provided ? inputs[1] : std::make_shared<NullNode>(),
                          axes,
                          inverse == 1,
                          onesided == 1)};
}
}  // namespace

namespace opset_1 {
ov::OutputVector dft(const ov::frontend::onnx::Node& node) {
    const auto axis = node.get_attribute_value<int64_t>("axis", 1);
    return translate_dft(node, ov::op::v0::Constant::create(ov::element::i64, {1}, {axis}));
}

ONNX_OP("DFT", OPSET_SINCE(1), ai_onnx::opset_1::dft);
}  // namespace opset_1

namespace opset_20 {
ov::OutputVector dft(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto i64 = ov::element::i64;
    const auto zero = ov::op::v0::Constant::create(i64, {1}, {0});
    ov::Output<ov::Node> axes;
    if (inputs.size() > 2 && !ov::op::util::is_null(inputs[2])) {
        axes = std::make_shared<ov::op::v0::Unsqueeze>(inputs[2], zero);
    } else {
        axes = ov::op::v0::Constant::create(i64, {1}, {-2});
    }

    // ONNX counts the trailing real/imaginary dimension in negative axes.
    const auto input_shape = std::make_shared<ov::op::v3::ShapeOf>(inputs.at(0));
    const auto rank = std::make_shared<ov::op::v3::ShapeOf>(input_shape);
    const auto adjusted = std::make_shared<ov::op::v1::Add>(axes, rank);
    const auto negative = std::make_shared<ov::op::v1::Less>(axes, zero);
    return translate_dft(node, std::make_shared<ov::op::v1::Select>(negative, adjusted, axes));
}

ONNX_OP("DFT", OPSET_SINCE(20), ai_onnx::opset_20::dft);
}  // namespace opset_20
}  // namespace ov::frontend::onnx::ai_onnx
