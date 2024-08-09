// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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
namespace ai_onnx {

// Link with an existing translator
namespace opset_1 {
extern ov::OutputVector identity(const ov::frontend::onnx::Node& node);
}  // namespace opset_1

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

const std::set<element::Type> supported_types_v1 =
    {element::u32, element::u64, element::i32, element::i64, element::f16, element::f32, element::f64};
const std::set<element::Type> supported_types_v2 =
    {element::u32, element::u64, element::i32, element::i64, element::f16, element::f32, element::f64, element::bf16};
const std::set<element::Type> supported_types_v3 = {element::u32,
                                                    element::u64,
                                                    element::i32,
                                                    element::i64,
                                                    element::f16,
                                                    element::f32,
                                                    element::f64,
                                                    element::bf16,
                                                    element::i8,
                                                    element::u8};
const std::set<element::Type> supported_types_v4 = {element::u32,
                                                    element::u64,
                                                    element::i32,
                                                    element::i64,
                                                    element::f16,
                                                    element::f32,
                                                    element::f64,
                                                    element::bf16,
                                                    element::i8,
                                                    element::u8,
                                                    element::boolean};

template <typename OpType>
std::shared_ptr<ov::Node> make_ov_reduction_op(const Node& node,
                                               const ov::Output<ov::Node>& ov_input,
                                               const std::set<element::Type>& supported_types,
                                               bool axes_as_attr = true) {
    const std::int64_t keepdims = node.get_attribute_value<std::int64_t>("keepdims", 1);

    CHECK_VALID_NODE(node,
                     supported_types.find(ov_input.get_element_type()) != supported_types.end(),
                     "Unsupported input type ",
                     ov_input.get_element_type().get_type_name());

    const auto reduction_axes = axes_as_attr ? get_reduction_axes_from_attr(node) : get_reduction_axes_from_input(node);
    if (reduction_axes != nullptr) {
        return std::make_shared<OpType>(ov_input, reduction_axes, static_cast<bool>(keepdims));
    } else {
        return ai_onnx::opset_1::identity(node).at(0).get_node_shared_ptr();
    }
}

std::shared_ptr<ov::Node> onnx_reduce_sum_square(const ov::frontend::onnx::Node& node,
                                                 const std::set<element::Type>& supported_types,
                                                 const bool axes_as_attr = true) {
    const auto input = ov::Output<ov::Node>{node.get_ov_inputs().at(0)};
    const auto square_node = std::make_shared<v1::Multiply>(input, input);
    return make_ov_reduction_op<v1::ReduceSum>(node, square_node, supported_types, axes_as_attr);
}
}  // namespace

namespace opset_1 {
ov::OutputVector reduce_log_sum(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> sum_node =
        make_ov_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0), supported_types_v2);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_log_sum_exp(const ov::frontend::onnx::Node& node) {
    const auto exp_node = std::make_shared<v0::Exp>(node.get_ov_inputs().at(0));
    const ov::Output<ov::Node> sum_node = make_ov_reduction_op<v1::ReduceSum>(node, exp_node, supported_types_v1);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_l1(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v4::ReduceL1>(node, node.get_ov_inputs().at(0), supported_types_v2)};
}

ov::OutputVector reduce_l2(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v4::ReduceL2>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMax>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_mean(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMean>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMin>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_prod(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceProd>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0), supported_types_v1)};
}

ov::OutputVector reduce_sum_square(const ov::frontend::onnx::Node& node) {
    return {onnx_reduce_sum_square(node, supported_types_v1)};
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("ReduceLogSum", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_log_sum);
    ONNX_OP_M("ReduceLogSumExp", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_log_sum_exp);
    ONNX_OP_M("ReduceL1", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_l1);
    ONNX_OP_M("ReduceL2", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_l2);
    ONNX_OP_M("ReduceMax", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_max);
    ONNX_OP_M("ReduceMean", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_mean);
    ONNX_OP_M("ReduceMin", {1, 12}, ai_onnx::opset_1::reduce_min);
    ONNX_OP_M("ReduceProd", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_prod);
    ONNX_OP_M("ReduceSum", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_sum);
    ONNX_OP_M("ReduceSumSquare", OPSET_RANGE(1, 12), ai_onnx::opset_1::reduce_sum_square);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1

/*
    Opset 11 is skipped because there are no significant difference between opset1 and opset 11.
    Found difference is:
    1. Operations (except ReduceMin and ReduceMax) are lost mention of zero-rank input behavior
       from their description. We assume it shouldn't be worse than opset 1.
    2. Opset 11 introduced requirement for axes values to be in a range [-r, r-1] where r = rank(data)
       Same time Reduce* operations in OpenVINO has same requirement from first version
*/

namespace opset_13 {
ov::OutputVector reduce_sum(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0), supported_types_v2, false)};
}

ov::OutputVector reduce_l2(const Node& node) {
    return {make_ov_reduction_op<v4::ReduceL2>(node, node.get_ov_inputs().at(0), supported_types_v2)};
}

ov::OutputVector reduce_log_sum_exp(const ov::frontend::onnx::Node& node) {
    const auto exp_node = std::make_shared<v0::Exp>(node.get_ov_inputs().at(0));
    const ov::Output<ov::Node> sum_node = make_ov_reduction_op<v1::ReduceSum>(node, exp_node, supported_types_v2);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMax>(node, node.get_ov_inputs().at(0), supported_types_v3)};
}

ov::OutputVector reduce_mean(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMean>(node, node.get_ov_inputs().at(0), supported_types_v2)};
}

ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMin>(node, node.get_ov_inputs().at(0), supported_types_v3)};
}

ov::OutputVector reduce_prod(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceProd>(node, node.get_ov_inputs().at(0), supported_types_v2)};
}

ov::OutputVector reduce_sum_square(const ov::frontend::onnx::Node& node) {
    return {onnx_reduce_sum_square(node, supported_types_v2)};
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("ReduceL2", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_l2);
    ONNX_OP_M("ReduceLogSumExp", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_log_sum_exp);
    ONNX_OP_M("ReduceMax", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_max);
    ONNX_OP_M("ReduceMean", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_mean);
    ONNX_OP_M("ReduceMin", {13, 17}, ai_onnx::opset_13::reduce_min);
    ONNX_OP_M("ReduceProd", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_prod);
    ONNX_OP_M("ReduceSum", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_sum);
    ONNX_OP_M("ReduceSumSquare", OPSET_RANGE(13, 17), ai_onnx::opset_13::reduce_sum_square);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_13

namespace opset_18 {
ov::OutputVector reduce_l2(const Node& node) {
    return {make_ov_reduction_op<v4::ReduceL2>(node, node.get_ov_inputs().at(0), supported_types_v2, false)};
}

ov::OutputVector reduce_log_sum_exp(const ov::frontend::onnx::Node& node) {
    const auto exp_node = std::make_shared<v0::Exp>(node.get_ov_inputs().at(0));
    const ov::Output<ov::Node> sum_node =
        make_ov_reduction_op<v1::ReduceSum>(node, exp_node, supported_types_v3, false);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMax>(node, node.get_ov_inputs().at(0), supported_types_v3, false)};
}

ov::OutputVector reduce_mean(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMean>(node, node.get_ov_inputs().at(0), supported_types_v3, false)};
}

ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceMin>(node, node.get_ov_inputs().at(0), supported_types_v3, false)};
}

ov::OutputVector reduce_log_sum(const ov::frontend::onnx::Node& node) {
    const ov::Output<ov::Node> sum_node =
        make_ov_reduction_op<v1::ReduceSum>(node, node.get_ov_inputs().at(0), supported_types_v2, false);
    return {std::make_shared<v0::Log>(sum_node)};
}

ov::OutputVector reduce_prod(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v1::ReduceProd>(node, node.get_ov_inputs().at(0), supported_types_v3, false)};
}

ov::OutputVector reduce_sum_square(const ov::frontend::onnx::Node& node) {
    return {onnx_reduce_sum_square(node, supported_types_v2, false)};
}

ov::OutputVector reduce_l1(const ov::frontend::onnx::Node& node) {
    return {make_ov_reduction_op<v4::ReduceL1>(node, node.get_ov_inputs().at(0), supported_types_v2, false)};
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("ReduceLogSum", OPSET_SINCE(18), ai_onnx::opset_18::reduce_log_sum);
    ONNX_OP_M("ReduceL2", OPSET_SINCE(18), ai_onnx::opset_18::reduce_l2);
    ONNX_OP_M("ReduceLogSumExp", OPSET_SINCE(18), ai_onnx::opset_18::reduce_log_sum_exp);
    ONNX_OP_M("ReduceMax", OPSET_RANGE(18, 19), ai_onnx::opset_18::reduce_max);
    ONNX_OP_M("ReduceMean", OPSET_SINCE(18), ai_onnx::opset_18::reduce_mean);
    ONNX_OP_M("ReduceMin", {18, 19}, ai_onnx::opset_18::reduce_min);
    ONNX_OP_M("ReduceProd", OPSET_SINCE(18), ai_onnx::opset_18::reduce_prod);
    ONNX_OP_M("ReduceSumSquare", OPSET_SINCE(18), ai_onnx::opset_18::reduce_sum_square);
    ONNX_OP_M("ReduceL1", OPSET_SINCE(18), ai_onnx::opset_18::reduce_l1);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_18

namespace opset_20 {
ov::OutputVector reduce_max(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    if (data.get_element_type() != element::boolean) {
        return {make_ov_reduction_op<v1::ReduceMax>(node, data, supported_types_v3, false)};
    } else {
        // Handling boolean as a uint8
        return {std::make_shared<v0::Convert>(
            make_ov_reduction_op<v1::ReduceMax>(node,
                                                std::make_shared<ov::op::v0::Convert>(data, element::u8),
                                                supported_types_v4,
                                                false),
            element::boolean)};
    }
}

ov::OutputVector reduce_min(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    if (data.get_element_type() != element::boolean) {
        return {make_ov_reduction_op<v1::ReduceMin>(node, data, supported_types_v3, false)};
    } else {
        // Handling boolean as a uint8
        return {std::make_shared<v0::Convert>(
            make_ov_reduction_op<v1::ReduceMin>(node,
                                                std::make_shared<ov::op::v0::Convert>(data, element::u8),
                                                supported_types_v4,
                                                false),
            element::boolean)};
    }
}

static bool register_multiple_translators(void) {
    ONNX_OP_M("ReduceMax", OPSET_SINCE(20), ai_onnx::opset_20::reduce_max);
    ONNX_OP_M("ReduceMin", OPSET_SINCE(20), ai_onnx::opset_20::reduce_min);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_20
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
