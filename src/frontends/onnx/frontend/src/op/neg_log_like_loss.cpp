// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/attribute.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

namespace detail {
static std::shared_ptr<ov::Node> get_dynamic_all_axes_range(const ov::Output<ov::Node>& input) {
    const auto shape_of_input = std::make_shared<v3::ShapeOf>(input);
    const auto scalar = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    const auto rank_of_input = std::make_shared<v3::ShapeOf>(shape_of_input);
    const auto rank_of_input_scalar = std::make_shared<v0::Squeeze>(rank_of_input, scalar);
    const auto start = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto step = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
    return std::make_shared<v4::Range>(start, rank_of_input_scalar, step, ov::element::i64);
}

ov::OutputVector negative_log_likelihood_loss(const ov::OutputVector inputs,
                                              const std::string reduction,
                                              bool use_ignore_index = false,
                                              const int64_t ignore_index_value = 0) {
    // Operator definition:
    // https://github.com/onnx/onnx/blob/a90ee0519933bd7412b04a3b7472eb550e78fcaf/onnx/defs/math/old.cc#L14
    const auto num_inputs = inputs.size();
    const auto& data = inputs[0];
    const auto& target = inputs[1];
    ov::Output<ov::Node> loss;

    const auto const_zero = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{0});
    const auto const_one = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{1});
    const auto axes = std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{1});

    const auto expanded_target = std::make_shared<v0::Unsqueeze>(target, axes);

    if (!use_ignore_index) {
        const auto gather_element = std::make_shared<v6::GatherElements>(data, expanded_target, 1);
        const auto loss_NCdd = std::make_shared<v0::Negative>(gather_element);
        const auto loss_N1dd = std::make_shared<v8::Slice>(loss_NCdd, const_zero, const_one, const_one, const_one);

        if (num_inputs < 3) {
            loss = std::make_shared<v0::Squeeze>(loss_N1dd, axes);
            const auto reduction_axes = get_dynamic_all_axes_range(loss);
            if (reduction == "mean") {
                loss = std::make_shared<v1::ReduceMean>(loss, reduction_axes);
            } else if (reduction == "sum") {
                loss = std::make_shared<v1::ReduceSum>(loss, reduction_axes);
            }
        } else {
            const auto& weights = inputs[2];
            const auto gather_weights = std::make_shared<v8::Gather>(weights, target, const_zero);
            const auto loss_unweighted = std::make_shared<v0::Squeeze>(loss_N1dd, axes);

            loss = std::make_shared<v1::Multiply>(loss_unweighted, gather_weights);
            if (reduction == "mean") {
                const auto loss_sum = std::make_shared<v1::ReduceSum>(loss, get_dynamic_all_axes_range(loss));
                const auto wg_sum =
                    std::make_shared<v1::ReduceSum>(gather_weights, get_dynamic_all_axes_range(gather_weights));
                loss = std::make_shared<v1::Divide>(loss_sum, wg_sum);
            } else if (reduction == "sum") {
                loss = std::make_shared<v1::ReduceSum>(loss, get_dynamic_all_axes_range(loss));
            }
        }
    } else {
        const auto const_ii =
            std::make_shared<v0::Constant>(ov::element::i64, Shape{1}, std::vector<int64_t>{ignore_index_value});

        const auto sub = std::make_shared<v1::Subtract>(expanded_target, expanded_target);

        const auto expanded_target_i64 = std::make_shared<v0::Convert>(expanded_target, ov::element::i64);
        const auto mask = std::make_shared<v1::Equal>(expanded_target_i64, const_ii);
        const auto transform_targets = std::make_shared<v1::Select>(mask, sub, expanded_target);

        const auto gather_element = std::make_shared<v6::GatherElements>(data, transform_targets, 1);

        ov::Output<ov::Node> input_gather_element_transform;

        const auto const_zero_converted = std::make_shared<v0::Convert>(const_zero, data.get_element_type());
        const auto const_one_converted = std::make_shared<v0::Convert>(const_one, data.get_element_type());

        input_gather_element_transform = std::make_shared<v1::Select>(mask, const_zero_converted, gather_element);

        const auto one_target = std::make_shared<v3::Broadcast>(const_one, std::make_shared<v0::ShapeOf>(const_one));
        const auto loss_NCdd = std::make_shared<v0::Negative>(input_gather_element_transform);
        const auto loss_N1dd = std::make_shared<v8::Slice>(loss_NCdd, const_zero, const_one, one_target, const_one);

        ov::Output<ov::Node> gather_weights;

        if (num_inputs < 3) {
            const auto squeeze_mask = std::make_shared<v0::Squeeze>(mask, axes);

            gather_weights = std::make_shared<v1::Select>(squeeze_mask, const_zero_converted, const_one_converted);
        } else {
            const auto& weights = inputs[2];
            const auto gather_weight_tmp = std::make_shared<v8::Gather>(weights, transform_targets, const_zero);
            gather_weights = std::make_shared<v1::Select>(mask, const_zero_converted, gather_weight_tmp);
            gather_weights = std::make_shared<v0::Squeeze>(gather_weights, axes);
        }

        const auto loss_unweighted = std::make_shared<v0::Squeeze>(loss_N1dd, axes);

        loss = std::make_shared<v1::Multiply>(loss_unweighted, gather_weights);
        if (reduction == "mean") {
            const auto loss_sum = std::make_shared<v1::ReduceSum>(loss, get_dynamic_all_axes_range(loss));
            const auto wg_sum =
                std::make_shared<v1::ReduceSum>(gather_weights, get_dynamic_all_axes_range(gather_weights));
            loss = std::make_shared<v1::Divide>(loss_sum, wg_sum);
        } else if (reduction == "sum") {
            loss = std::make_shared<v1::ReduceSum>(loss, get_dynamic_all_axes_range(loss));
        }
    }

    return {loss};
}
}  // namespace detail

ov::OutputVector negative_log_likelihood_loss(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 2);
    const auto inputs = node.get_ov_inputs();

    const auto reduction = node.get_attribute_value<std::string>("reduction", "mean");
    bool ignore_index = node.has_attribute("ignore_index") == true;
    int64_t ignore_index_value = 0;

    // In some cases attributemay exists, but has an "undefined" type, which means for us "not exist"
    if (ignore_index) {
        const auto& attr = node.get_attribute("ignore_index");
        ignore_index = attr.get_type() != Attribute::Type::undefined;
        if (ignore_index) {
            ignore_index_value = node.get_attribute_value<int64_t>("ignore_index");
        }
    }

    CHECK_VALID_NODE(node,
                     reduction == "none" || reduction == "sum" || reduction == "mean",
                     "NegativeLogLikelihoodLoss expects reduction: none, sum, mean. Got: ",
                     reduction);

    return detail::negative_log_likelihood_loss(inputs, reduction, ignore_index, ignore_index_value);
}

ONNX_OP("NegativeLogLikelihoodLoss", OPSET_SINCE(1), ai_onnx::opset_1::negative_log_likelihood_loss);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
