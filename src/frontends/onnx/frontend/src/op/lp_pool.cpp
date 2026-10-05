// Copyright (C) 2018-2026 Intel Corporation
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
#include "utils/pooling_factory.hpp"
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
    FRONT_END_GENERAL_CHECK(data_shape[channel_axis].is_static(), "Channel dimension of input data must be static");

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

ov::OutputVector lp_pool(const ov::frontend::onnx::Node& node) {
    // In opset 1 the 'p' attribute is a float, since opset 2 it is an integer.
    const auto p_norm = node.get_attribute_value<float>("p", 2.f);
    return pooling::PoolingFactory(node).make_lp_pool(p_norm);
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("GlobalLpPool", OPSET_SINCE(1), ai_onnx::opset_1::global_lp_pool);
    ONNX_OP_M("LpPool", OPSET_IN(1), ai_onnx::opset_1::lp_pool);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1

namespace opset_2 {
ov::OutputVector lp_pool(const ov::frontend::onnx::Node& node) {
    const auto p_norm = node.get_attribute_value<std::int64_t>("p", 2);
    return pooling::PoolingFactory(node).make_lp_pool(static_cast<float>(p_norm));
}

ONNX_OP("LpPool", OPSET_SINCE(2), ai_onnx::opset_2::lp_pool);
}  // namespace opset_2
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
