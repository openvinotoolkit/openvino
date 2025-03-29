// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "utils/common.hpp"
#include "utils/norm.hpp"
#include "utils/split.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector global_lp_pool(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> data{node.get_ov_inputs().at(0)};
    const std::size_t channel_axis{1};

    const auto data_shape = data.get_partial_shape();
    FRONT_END_GENERAL_CHECK(data_shape.rank().is_static(), "Rank of input data must be static");
    FRONT_END_GENERAL_CHECK(data_shape.rank().get_length() >= 2, "Rank of input data must be greater or equal to 2");
    FRONT_END_GENERAL_CHECK(data_shape[0].is_static(), "First dimension of input data must be static");
    FRONT_END_GENERAL_CHECK(data_shape[channel_axis].is_static(), "Channel dimension of intput data must be static");

    const std::size_t channels_count = data_shape[channel_axis].get_length();
    const std::int64_t p_norm{node.get_attribute_value<std::int64_t>("p", 2)};

    CHECK_VALID_NODE(node, p_norm >= 0, "Only positive (including zero) values are supported for 'p' attribute.");

    ov::OutputVector slices = ov::op::util::make_split(data, channels_count, channel_axis);

    for (auto& slice : slices) {
        // all dimensions except spatial/feature
        const auto reduction_axes = common::get_monotonic_range_along_node_rank(data, 2);

        slice = ov::op::util::lp_norm(slice, reduction_axes, static_cast<std::size_t>(p_norm));

        // output shape is all ones except N channel
        ov::Shape output_shape(data_shape.rank().get_length(), 1);
        output_shape.at(0) = data_shape[0].get_length();

        const auto reshape_pattern =
            v0::Constant::create(ov::element::i64, ov::Shape{output_shape.size()}, output_shape);

        slice = std::make_shared<v1::Reshape>(slice, reshape_pattern, false);
    }

    return {std::make_shared<v0::Concat>(slices, channel_axis)};
}

ONNX_OP("GlobalLpPool", OPSET_SINCE(1), ai_onnx::opset_1::global_lp_pool);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
