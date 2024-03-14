// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/reduce.hpp"

#include "exceptions.hpp"
#include "identity.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_l1.hpp"
#include "openvino/op/reduce_l2.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_min.hpp"
#include "openvino/op/reduce_prod.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace {
std::shared_ptr<ov::Node> get_dynamic_all_axes_range(const Node& node) {
    const auto input = node.get_ov_inputs().at(0);
    const auto shape_of_input = std::make_shared<v3::ShapeOf>(input);
    const auto scalar = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    const auto rank_of_input = std::make_shared<v3::ShapeOf>(shape_of_input);
    const auto rank_of_input_scalar = std::make_shared<v0::Squeeze>(rank_of_input, scalar);
    const auto start = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto step = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    return std::make_shared<v4::Range>(start, rank_of_input_scalar, step, ov::element::i64);
}

std::shared_ptr<ov::Node> get_reduction_axes_from_input(const Node& node) {
    const std::int64_t noop_with_empty_axes = node.get_attribute_value<std::int64_t>("noop_with_empty_axes", 0);
    const auto input = node.get_ov_inputs().at(0);
    if (node.get_ov_inputs().size() > 1) {
        const auto reduction_axes = node.get_ov_inputs().at(1);
        const auto reduction_axes_rank = reduction_axes.get_partial_shape().rank();
        FRONT_END_GENERAL_CHECK(reduction_axes.get_partial_shape().is_static(),
                                "The axes tensor's shape needs to be known(static). Node: ",
                                node.get_description());

        if (reduction_axes_rank.get_length() != 0 && reduction_axes.get_shape() != ov::Shape{0}) {
            return reduction_axes.get_node_shared_ptr();
        }
    }

    if (noop_with_empty_axes) {
        return nullptr;
    } else {
        return get_dynamic_all_axes_range(node);
    }
}

std::shared_ptr<ov::Node> get_reduction_axes_from_attr(const Node& node) {
    auto reduction_axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {});

    const auto input_rank = node.get_ov_inputs().at(0).get_partial_shape().rank();

    if (reduction_axes.empty()) {
        if (input_rank.is_static()) {
            reduction_axes = ov::frontend::onnx::common::get_monotonic_range<int64_t>(input_rank.get_length());
        } else {
            return get_dynamic_all_axes_range(node);
        }
    }

    if (input_rank.is_static()) {
        CHECK_VALID_NODE(node,
                         static_cast<int64_t>(reduction_axes.size()) <= input_rank.get_length(),
                         "Number of reduction axes (",
                         reduction_axes.size(),
                         ") is larger than the input tensor's rank (",
                         input_rank.get_length(),
                         ")");
    }

    return v0::Constant::create(ov::element::i64, ov::Shape{reduction_axes.size()}, reduction_axes);
}

template <typename OpType>
std::shared_ptr<ov::Node> make_ng_reduction_op(const Node& node,
                                               const ov::Output<ov::Node>& ng_input,
                                               bool axes_as_attr = true) {
    const std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);

    const auto reduction_axes = axes_as_attr ? get_reduction_axes_from_attr(node) : get_reduction_axes_from_input(node);
    if (reduction_axes != nullptr) {
        return std::make_shared<OpType>(ng_input, reduction_axes, static_cast<bool>(keepdims));
    } else {
        return set_1::identity(node).at(0).get_node_shared_ptr();
    }
}
}  // namespace

namespace set_13 {
ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0), false)};
}
}  // namespace set_13

namespace set_1 {
ov::OutputVector reduce_log_sum(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> sum_node = make_ng_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0));
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_log_sum_exp(const ov::frontend::onnx::Node& node) {
    const auto exp_node = std::make_shared<v0::Exp>(node.get_ov_inputs().at(0));
    const ov::Output<ov::Node> sum_node = make_ng_reduction_op<v1::ReduceSum>(node, exp_node);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_l1(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v4::ReduceL1>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_l2(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v4::ReduceL2>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceMax>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_mean(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceMean>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceMin>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_prod(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceProd>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node) {
    return {make_ng_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0))};
}

ov::OutputVector reduce_sum_square(const ov::frontend::onnx::Node& node) {
    const auto input = ov::Output<ov::Node>{node.get_ov_inputs().at(0)};
    const auto square_node = std::make_shared<v1::Multiply>(input, input);
    return {make_ng_reduction_op<v1::ReduceSum>(node, square_node)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
