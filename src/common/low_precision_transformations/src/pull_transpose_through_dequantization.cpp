// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/pull_transpose_through_dequantization.hpp"

#include <assert.h>
#include <memory>
#include <queue>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

using namespace ngraph;

namespace pull_transpose_through_dequantization {
namespace {

std::shared_ptr<Node> moveThroughElementwise(const std::shared_ptr<Node>& transpose, const std::shared_ptr<Node>& elementwise) {
    const auto transposeValues = transpose->get_input_node_shared_ptr(1);
    NGRAPH_CHECK(transposeValues != nullptr, "transpose constant was not found");

    auto elementwiseValuesConvert = ov::as_type_ptr<opset1::Convert>(elementwise->get_input_node_shared_ptr(1ul));
    auto elementwiseValues = elementwiseValuesConvert == nullptr ?
        elementwise->get_input_node_shared_ptr(1ul) :
        elementwiseValuesConvert->get_input_node_shared_ptr(0ul);
    assert(ov::is_type<opset1::Constant>(elementwiseValues));

    const auto transposeValuesShape = transposeValues->get_output_shape(0);
    const auto elementwiseValuesShape = elementwiseValues->get_output_shape(0);
    if (elementwiseValuesShape.size() != shape_size(transposeValuesShape)) {
        if (shape_size(elementwiseValuesShape) != 1ul) {
            return nullptr;
        }

        elementwiseValues = ngraph::pass::low_precision::fold<opset1::Broadcast>(
            elementwiseValues,
            std::make_shared<opset1::Constant>(
                element::i64,
                Shape{ shape_size(transposeValuesShape) },
                std::vector<size_t>(shape_size(transposeValuesShape), 1ul)));
        assert(ov::is_type<opset1::Constant>(elementwiseValues));
    }

    const std::shared_ptr<opset1::Transpose> newTranspose = ov::as_type_ptr<opset1::Transpose>(transpose->clone_with_new_inputs({
        elementwise->get_input_node_shared_ptr(0ul),
        transposeValues }));

    const auto newElementwiseValues = ngraph::pass::low_precision::fold<opset1::Transpose>(
        elementwiseValues,
        transposeValues);
    assert(ov::is_type<opset1::Constant>(newElementwiseValues));

    const auto newElementwise = elementwise->clone_with_new_inputs({
        newTranspose,
        elementwiseValuesConvert == nullptr ?
            newElementwiseValues :
            std::make_shared<opset1::Convert>(newElementwiseValues, elementwiseValuesConvert->get_destination_type()) });

    replace_node(transpose, newElementwise);
    copy_runtime_info({ elementwise, transpose }, { newTranspose, newElementwise });

    return newTranspose;
}

std::shared_ptr<Node> moveThroughConvert(const std::shared_ptr<Node>& transpose, const std::shared_ptr<Node>& convert) {
    const auto newTranspose = transpose->clone_with_new_inputs({convert->input_value(0), transpose->input_value(1) });
    const auto newConvert = convert->clone_with_new_inputs({ newTranspose });
    replace_node(transpose, newConvert);
    copy_runtime_info({ convert, transpose }, { newTranspose, newConvert });

    return newTranspose;
}

void fuseConstant(const std::shared_ptr<Node>& transpose, const std::shared_ptr<Node>& constant) {
    const auto newConstant = ngraph::pass::low_precision::fold<opset1::Transpose>(
        constant,
        transpose->input_value(1));

    replace_node(transpose, newConstant);
    copy_runtime_info({ constant, transpose }, newConstant);
}

}  // namespace
}  // namespace pull_transpose_through_dequantization

ngraph::pass::low_precision::PullTransposeThroughDequantization::PullTransposeThroughDequantization(
    const std::vector<ngraph::element::Type>& inputPrecisions) {
    const auto weights = ngraph::pattern::wrap_type<ngraph::opset1::Constant>(pattern::type_matches_any(inputPrecisions));
    const auto convert = ngraph::pattern::wrap_type<ngraph::opset1::Convert>({ weights });

    MATCHER_SCOPE(PullTransposeThroughDequantization);
    const auto subtractValues = std::make_shared<pattern::op::Or>(OutputVector{
        ngraph::pattern::wrap_type<ngraph::opset1::Constant>(),
        ngraph::pattern::wrap_type<ngraph::opset1::Convert>({ngraph::pattern::wrap_type<ngraph::opset1::Constant>()})
    });
    const auto subtract = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>({ convert, subtractValues });

    const auto subtractOrConvert = std::make_shared<pattern::op::Or>(OutputVector{ convert, subtract });

    const auto multiplyConstant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    const auto multiply = ngraph::pattern::wrap_type<ngraph::opset1::Multiply>({ subtractOrConvert, multiplyConstant });

    const auto transposeConstant = ngraph::pattern::wrap_type<ngraph::opset1::Constant>();
    auto matcherTranspose = ngraph::pattern::wrap_type<opset1::Transpose>({ multiply, transposeConstant });

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher & m) -> bool {
        const auto& opsMap = m.get_pattern_value_map();
        auto transpose = opsMap.find(matcherTranspose)->second.get_node()->shared_from_this();

        while (transpose != nullptr) {
            const auto parent = transpose->get_input_node_shared_ptr(0);
            if (ov::is_type<opset1::Multiply>(parent) || ov::is_type<opset1::Subtract>(parent)) {
                transpose = pull_transpose_through_dequantization::moveThroughElementwise(transpose, parent);
            } else if (ov::is_type<opset1::Convert>(parent)) {
                transpose = pull_transpose_through_dequantization::moveThroughConvert(transpose, parent);
            } else if (ov::is_type<opset1::Constant>(parent)) {
                pull_transpose_through_dequantization::fuseConstant(transpose, ov::as_type_ptr<opset1::Constant>(parent));
                transpose = nullptr;
            } else {
                THROW_IE_LPT_EXCEPTION(*parent) << "unexepcted operation type";
            }
        }
        MATCHER_SCOPE_ENABLE(PullTransposeThroughDequantization);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcherTranspose, matcher_name);
    this->register_matcher(m, callback);
}
