// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace {
std::shared_ptr<ov::Node> get_filter_zero_point(const ov::OutputVector& inputs) {
    const auto& original_zero_point =
        (inputs.size() > 3) ? inputs.at(3) : v0::Constant::create(ov::element::i32, {}, {0});

    const auto filter_zero_point_rank = original_zero_point.get_partial_shape().rank();
    if (filter_zero_point_rank.is_static() && filter_zero_point_rank.get_length() == 0) {
        return std::make_shared<v0::Convert>(original_zero_point, ov::element::i32);
    } else {
        // in case of 1D zero point filter, it has to be unsqueezed to match the data input's rank
        const auto& converted_filter_zero_point = std::make_shared<v0::Convert>(original_zero_point, ov::element::i32);
        const auto& input_shape = std::make_shared<v3::ShapeOf>(inputs.at(0), ov::element::i32);
        const auto& input_rank = std::make_shared<v3::ShapeOf>(input_shape, ov::element::i32);
        const auto& input_rank_scalar = reshape::interpret_as_scalar(input_rank);

        const auto& one_node = v0::Constant::create(ov::element::i32, {}, {1});
        const auto& missing_dimensions =
            std::make_shared<v4::Range>(one_node, input_rank_scalar, one_node, ov::element::i32);

        return std::make_shared<v0::Unsqueeze>(converted_filter_zero_point, missing_dimensions);
    }
}
}  // namespace
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector conv_integer(const ov::frontend::onnx::Node& node) {
    const ov::OutputVector& inputs = node.get_ov_inputs();

    const auto& input = inputs.at(0);
    const auto& filter = inputs.at(1);
    const auto& input_zero_point = (inputs.size() > 2) ? inputs.at(2) : v0::Constant::create(ov::element::i32, {}, {0});

    const auto& converted_input = std::make_shared<v0::Convert>(input, ov::element::i32);
    const auto& converted_filter = std::make_shared<v0::Convert>(filter, ov::element::i32);

    const auto& converted_input_zero_point = std::make_shared<v0::Convert>(input_zero_point, ov::element::i32);
    const auto& filter_zero_point = get_filter_zero_point(inputs);

    const auto& shifted_input = std::make_shared<v1::Subtract>(converted_input, converted_input_zero_point);
    const auto& shifted_filter = std::make_shared<v1::Subtract>(converted_filter, filter_zero_point);

    const auto& groups = node.get_attribute_value<int64_t>("group", 1);
    const auto& strides = convpool::get_strides(node);
    const auto& dilations = convpool::get_dilations(node);
    const auto& paddings = convpool::get_pads(node);
    const ov::op::PadType& auto_pad_type = convpool::get_auto_pad(node);
    const auto& padding_below = paddings.first;
    const auto& padding_above = paddings.second;

    const auto conv_node = conv_factory::make_ng_convolution(shifted_input,
                                                             shifted_filter,
                                                             strides,
                                                             dilations,
                                                             padding_below,
                                                             padding_above,
                                                             groups,
                                                             auto_pad_type);

    return {conv_node};
}
ONNX_OP("ConvInteger", OPSET_SINCE(1), ai_onnx::opset_1::conv_integer);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
