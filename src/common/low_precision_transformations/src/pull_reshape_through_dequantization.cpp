// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/pull_reshape_through_dequantization.hpp"

#include <memory>
#include <queue>
#include <vector>

#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

using namespace ov;

namespace pull_reshape_through_dequantization {
namespace {

std::shared_ptr<Node> moveThroughElementwise(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& elementwise) {
    const auto reshapeValues = ov::as_type_ptr<opset1::Constant>(reshape->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(reshapeValues != nullptr, "Reshape constant was not found");

    auto elementwiseValuesConvert = ov::as_type_ptr<opset1::Convert>(elementwise->get_input_node_shared_ptr(1));
    auto elementwiseValues = elementwiseValuesConvert == nullptr ?
        ov::as_type_ptr<opset1::Constant>(elementwise->get_input_node_shared_ptr(1)) :
        ov::as_type_ptr<opset1::Constant>(elementwiseValuesConvert->get_input_node_shared_ptr(0));
    assert(elementwiseValues != nullptr);

    const auto newElementwiseValues = [&]() -> std::shared_ptr<Node> {
        // Firstly is checked whether the result constant shape can be set without any calculations
        const auto& elementwiseValuesShape = elementwiseValues->get_shape();
        if (ov::shape_size(elementwiseValuesShape) == 1) {
            return std::make_shared<opset1::Constant>(*elementwiseValues, Shape{});
        }

        const auto& targetShape = reshape->get_output_shape(0);
        if (targetShape.size() == elementwiseValuesShape.size()) {
            bool eltwiseConstAffected = false;
            for (size_t i = 0; i < targetShape.size(); ++i) {
                if (elementwiseValuesShape[i] != 1 && elementwiseValuesShape[i] != targetShape[i]) {
                    eltwiseConstAffected = true;
                    break;
                }
            }
            if (!eltwiseConstAffected) {
                return elementwiseValues;
            }
        }

        // If shape calculation is needed, fold_reshape is used for result constant shape computation
        const auto newReshapeValuesVector = ov::pass::low_precision::NetworkHelper::updateReshapeValues(
            elementwiseValuesShape,
            elementwise->get_output_shape(0),
            targetShape);

        // in some cases it's impossible to definitely reshape eltwise constant using Reshape
        if (ov::shape_size(newReshapeValuesVector) != ov::shape_size(elementwiseValuesShape)) {
            return nullptr;
        }

        const auto newReshapeValues = std::make_shared<opset1::Constant>(
            reshapeValues->get_output_element_type(0),
            Shape{ newReshapeValuesVector.size() },
            newReshapeValuesVector);

        const auto newElementwiseValues = ov::pass::low_precision::fold_reshape<opset1::Reshape>(
            elementwiseValues,
            newReshapeValues,
            ov::as_type_ptr<opset1::Reshape>(reshape)->get_special_zero());
        assert(ov::is_type<opset1::Constant>(newElementwiseValues));
        return newElementwiseValues;
    }();

    if (newElementwiseValues == nullptr) {
        return nullptr;
    }

    const auto newReshape = reshape->clone_with_new_inputs({elementwise->input_value(0), reshapeValues});
    const auto newElementwise = elementwise->clone_with_new_inputs({
        newReshape,
        elementwiseValuesConvert == nullptr ?
            newElementwiseValues :
            std::make_shared<opset1::Convert>(newElementwiseValues, elementwiseValuesConvert->get_destination_type()) });

    replace_node(reshape, newElementwise);
    ov::copy_runtime_info({ elementwise, reshape }, { newReshape, newElementwise });
    return newReshape;
}

std::shared_ptr<Node> moveThroughConvert(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& convert) {
    const auto newReshape = reshape->clone_with_new_inputs({ convert->input_value(0), reshape->input_value(1) });
    const auto newConvert = convert->clone_with_new_inputs({ newReshape });
    replace_node(reshape, newConvert);
    ov::copy_runtime_info({ convert, reshape }, { newReshape, newConvert });

    return newReshape;
}

void fuseConstant(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& constant) {
    ov::OutputVector result(1);
    OPENVINO_ASSERT(reshape->constant_fold(result, { constant, reshape->input_value(1) }), "Reshape constant folding failed");
    const auto newConstant = result[0].get_node_shared_ptr();
    replace_node(reshape, newConstant);
    copy_runtime_info({ constant, reshape }, newConstant);
}

}  // namespace
}  // namespace pull_reshape_through_dequantization

ov::pass::low_precision::PullReshapeThroughDequantization::PullReshapeThroughDequantization(
    const std::vector<ov::element::Type>& inputPrecisions) {
    const auto weights = ov::pass::pattern::wrap_type<ov::opset1::Constant>(pattern::type_matches_any(inputPrecisions));
    const auto convert = ov::pass::pattern::wrap_type<ov::opset1::Convert>({ weights });

    MATCHER_SCOPE(PullReshapeThroughDequantization);
    const auto subtractValues = std::make_shared<pass::pattern::op::Or>(OutputVector{
        ov::pass::pattern::wrap_type<ov::opset1::Constant>(),
        ov::pass::pattern::wrap_type<ov::opset1::Convert>({ov::pass::pattern::wrap_type<ov::opset1::Constant>()})
    });
    const auto subtract = ov::pass::pattern::wrap_type<ov::opset1::Subtract>({ convert, subtractValues });

    const auto subtractOrConvert = std::make_shared<pass::pattern::op::Or>(OutputVector{ convert, subtract });

    const auto multiplyConstant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    const auto multiply = ov::pass::pattern::wrap_type<ov::opset1::Multiply>({ subtractOrConvert, multiplyConstant });

    const auto reshapeConstant = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    auto reshapeWrapper = ov::pass::pattern::wrap_type<opset1::Reshape>({ multiply, reshapeConstant });

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        auto reshape = opsMap.at(reshapeWrapper).get_node_shared_ptr();

        auto child = reshape->get_output_target_inputs(0).begin()->get_node();
        if (ov::is_type<opset1::GroupConvolution>(child)) {
            return false;
        }

        do {
            const auto parent = reshape->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Multiply>(parent) || ov::is_type<opset1::Subtract>(parent)) {
                reshape = pull_reshape_through_dequantization::moveThroughElementwise(reshape, parent);
            } else if (ov::is_type<opset1::Convert>(parent)) {
                reshape = pull_reshape_through_dequantization::moveThroughConvert(reshape, parent);
            } else if (ov::is_type<opset1::Constant>(parent)) {
                pull_reshape_through_dequantization::fuseConstant(reshape, ov::as_type_ptr<opset1::Constant>(parent));
                reshape = nullptr;
            } else {
                THROW_IE_LPT_EXCEPTION(*parent) << "unexepcted operation type";
            }
        } while (reshape != nullptr);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(reshapeWrapper, matcher_name);
    this->register_matcher(m, callback);
}
