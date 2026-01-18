// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/col2im.hpp"

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/validation_util.hpp"
#include "utils/common.hpp"
using namespace ov::op::v15;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_18 {
ov::OutputVector col2im(const ov::frontend::onnx::Node& node) {
    // 1. get inputs
    common::default_op_checks(node, 3);
    const auto inputs = node.get_ov_inputs();
    const auto& data = inputs[0];         // input
    const auto& output_size = inputs[1];  // image_shape
    const auto& kernel_size = inputs[2];  // block_shape

    // 2. get attributes
    const size_t spatial_rank = 2;

    std::vector<size_t> default_attr_vals(spatial_rank, 1);
    auto dilations = node.get_attribute_value<std::vector<size_t>>("dilations", default_attr_vals);
    auto strides = node.get_attribute_value<std::vector<size_t>>("strides", default_attr_vals);

    std::vector<size_t> default_pads(spatial_rank * 2, 0);
    auto pads = node.get_attribute_value<std::vector<size_t>>("pads", default_pads);
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;

    // ov::op::v15::Col2Im only supports 2D spatial dimensions
    CHECK_VALID_NODE(node,
                     dilations.size() == spatial_rank,
                     "Col2Im 'dilations' attribute must have size [2]. Got: ",
                     dilations.size());
    CHECK_VALID_NODE(node,
                     strides.size() == spatial_rank,
                     "Col2Im 'strides' attribute must have size [2]. Got: ",
                     strides.size());
    CHECK_VALID_NODE(node,
                     pads.size() == spatial_rank * 2,
                     "Col2Im 'pads' attribute must have size [4]. Got: ",
                     pads.size());

    // spatial_rank == pads.size() / 2
    pads_begin.assign(pads.begin(), pads.begin() + spatial_rank);
    pads_end.assign(pads.begin() + spatial_rank, pads.end());

    // 3. return Col2Im
    return {
        std::make_shared<ov::op::v15::Col2Im>(data, output_size, kernel_size, strides, dilations, pads_begin, pads_end)
            ->outputs()};
}

ONNX_OP("Col2Im", OPSET_SINCE(1), ai_onnx::opset_18::col2im);
}  // namespace opset_18
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
