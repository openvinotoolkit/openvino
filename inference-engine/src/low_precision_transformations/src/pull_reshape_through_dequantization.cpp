// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/pull_reshape_through_dequantization.hpp"

#include <memory>
#include <queue>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::PullReshapeThroughDequantization, "PullReshapeThroughDequantizationFusion", 0);

namespace pull_reshape_through_dequantization {

std::shared_ptr<Node> moveThroughElementwise(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& elementwise) {
    const auto reshapeValues = reshape->get_input_node_shared_ptr(1);
    NGRAPH_CHECK(reshapeValues != nullptr, "Reshape constant was not found");

    auto elementwiseValuesConvert = as_type_ptr<opset1::Convert>(elementwise->get_input_node_shared_ptr(1ul));
    auto elementwiseValues = elementwiseValuesConvert == nullptr ?
        elementwise->get_input_node_shared_ptr(1ul) :
        elementwiseValuesConvert->get_input_node_shared_ptr(0ul);
    assert(is_type<opset1::Constant>(elementwiseValues));

    const std::shared_ptr<opset1::Reshape> newReshape = as_type_ptr<opset1::Reshape>(reshape->clone_with_new_inputs({
        elementwise->get_input_node_shared_ptr(0ul),
        reshapeValues }));

    std::shared_ptr<Node> newElementwiseValues;

    const Shape elementwiseValuesShape = elementwiseValues->output(0).get_shape();
    if (!elementwiseValuesShape.empty() && (elementwiseValuesShape.size() != 1ul)) {
        // update shape constant value to avoid eltwise constan value broadcasting
        const Shape elementwiseShape = elementwise->output(0).get_shape();
        const std::vector<size_t> reshapeValuesVector = as_type_ptr<opset1::Constant>(reshapeValues)->cast_vector<size_t>();

        const std::vector<size_t> newReshapeValuesVector = ngraph::pass::low_precision::NetworkHelper::updateReshapeValues(
            elementwiseValuesShape,
            elementwiseShape,
            reshapeValuesVector);

        const auto newReshapeValues = std::make_shared<opset1::Constant>(
            reshapeValues->output(0).get_element_type(),
            Shape{ newReshapeValuesVector.size() },
            newReshapeValuesVector);

        newElementwiseValues = ngraph::pass::low_precision::fold_reshape<opset1::Reshape>(
            elementwiseValues->output(0),
            newReshapeValues->output(0),
            as_type_ptr<opset1::Reshape>(reshape)->get_special_zero());
        assert(is_type<opset1::Constant>(newElementwiseValues));
    } else {
        newElementwiseValues = elementwiseValues;
    }
    const auto newElementwise = elementwise->clone_with_new_inputs({
        newReshape,
        elementwiseValuesConvert == nullptr ?
            newElementwiseValues :
            std::make_shared<opset1::Convert>(newElementwiseValues, elementwiseValuesConvert->get_destination_type()) });

    replace_node(reshape, newElementwise);
    copy_runtime_info({ elementwise, reshape }, { newReshape, newElementwise });
    return newReshape;
}

std::shared_ptr<Node> moveThroughConvert(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& convert) {
    const auto newReshape = reshape->clone_with_new_inputs({ convert->get_input_node_shared_ptr(0), reshape->get_input_node_shared_ptr(1) });
    const auto newConvert = convert->clone_with_new_inputs({ newReshape });
    replace_node(reshape, newConvert);
    copy_runtime_info({ convert, reshape }, { newReshape, newConvert });

    return newReshape;
}

void fuseConstant(const std::shared_ptr<Node>& reshape, const std::shared_ptr<Node>& constant) {
    ngraph::OutputVector result(1);
    reshape->constant_fold(result, { constant->output(0), reshape->get_input_node_ptr(1)->output(0) });
    const auto newConstant = result[0].get_node_shared_ptr();
    replace_node(reshape, newConstant);
    copy_runtime_info({ constant, reshape }, newConstant);
}

}  // namespace pull_reshape_through_dequantization

ngraph::pass::low_precision::PullReshapeThroughDequantization::PullReshapeThroughDequantization(
    const std::vector<ngraph::element::Type>& inputPrecisions) {
    const auto weights = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(pattern::type_matches_any(inputPrecisions));
    const auto convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({ weights });

    const auto subtractValues = std::make_shared<pattern::op::Or>(OutputVector{
        ngraph::pattern::wrap_type<ngraph::opset1::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset1::Convert>({ngraph::pattern::wrap_type<ngraph::opset1::Constant>()})
    });
    const auto subtract = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>({ convert, subtractValues });

    const auto subtractOrConvert = std::make_shared<pattern::op::Or>(OutputVector{ convert, subtract });

    const auto multiplyConstant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto multiply = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>({ subtractOrConvert, multiplyConstant });

    const auto reshapeConstant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto reshapeWrapper = ngraph::pattern::wrap_type<opset1::Reshape>({ multiply, reshapeConstant });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        auto reshape = opsMap.find(reshapeWrapper)->second.get_node()->shared_from_this();

        auto child = reshape->get_output_target_inputs(0).begin()->get_node();
        if (is_type<opset1::GroupConvolution>(child)) {
            return false;
        }

        while (reshape != nullptr) {
            const auto parent = reshape->get_input_node_shared_ptr(0);
            if (is_type<opset1::Multiply>(parent) || is_type<opset1::Subtract>(parent)) {
                reshape = pull_reshape_through_dequantization::moveThroughElementwise(reshape, parent);
            } else if (is_type<opset1::Convert>(parent)) {
                reshape = pull_reshape_through_dequantization::moveThroughConvert(reshape, parent);
            } else if (is_type<opset1::Constant>(parent)) {
                pull_reshape_through_dequantization::fuseConstant(reshape, as_type_ptr<opset1::Constant>(parent));
                reshape = nullptr;
            } else {
                THROW_IE_LPT_EXCEPTION(*parent) << "unexepcted operation type";
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(reshapeWrapper, "PullReshapeThroughDequantization");
    this->register_matcher(m, callback);
}
