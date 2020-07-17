// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "ngraph_ops/type_relaxed.hpp"

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void AddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    // addPattern(
    //        pass,
    //        context,
    //        make_op_pattern<opset1::Add>(
    //                { make_op_label<opset1::Multiply>(),
    //                  make_op_label<opset1::Constant>()}));
    // addPattern(
    //        pass,
    //        context,
    //        make_op_pattern<opset1::Add>(
    //                { make_op_label<opset1::Constant>(),
    //                  make_op_label<opset1::Multiply>() }));

    addSingleNodePattern<opset1::Add>(pass, context);
}


void AddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // TODO: move to handler
    if (!canBeTransformed(context, m.get_match_root())) {
        return;
    }

    auto add = m.get_match_root();

    add = separateInStandaloneBranch(add);

    const int fullPathIndex = getNotEmpty(add);

    std::shared_ptr<opset1::Multiply> newMultiply;

    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(add);

        if (multiplyBranch.first == -1 || multiplyBranch.second == -1)
            return;

        newMultiply = NetworkHelper::swapMultiplyAndAdd(add, multiplyBranch);
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        // TODO: question: is it reasonable to create Constant? (performance issue?)
        // TODO: question: should we clone constant here?

        FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::getDequantization(add, emptyPathIndex);
        std::shared_ptr<Node> subtractEmptyPathValues;
        std::shared_ptr<Node> multiplyEmptyPathValues;
        std::tie(subtractEmptyPathValues, multiplyEmptyPathValues) = NetworkHelper::createEmptyValues(dequantizationEmptyPath);

        FakeQuantizeDequantization dequantizationFullPath = NetworkHelper::getDequantization(add, fullPathIndex);
        std::shared_ptr<Node> subtractFullPathValues;
        std::shared_ptr<Node> multiplyFullPathValues;
        std::tie(subtractFullPathValues, multiplyFullPathValues) = NetworkHelper::createEmptyValues(dequantizationFullPath);

        // calculation
        // before: Y = (SC1 * (X1 - SH1)) + (SC2 * (X2 - SH2))
        // after : Y = SC2 * ( SC1' * (X1 - SH1') + X2 ) , where :
        //         SC1' = SC1 / SC2
        //         SH1' = SH1 + SC2 * SH2 / SC1
        std::shared_ptr<Node> newSubtractFullPathValues = fold<opset1::Add>(
            subtractFullPathValues,
            fold<opset1::Divide>(
                fold<opset1::Multiply>(subtractEmptyPathValues, multiplyEmptyPathValues),
                multiplyFullPathValues));

        std::shared_ptr<Node> newMultiplyFullPathValues = fold<opset1::Divide>(multiplyFullPathValues, multiplyEmptyPathValues);

        if (NetworkHelper::isZeroConst(newSubtractFullPathValues)) {
            newSubtractFullPathValues = nullptr;
        }

        // graph update
        std::vector<std::shared_ptr<Node>> inputs{ {}, {} };
        auto fullPathInput = dequantizationFullPath.convert == nullptr ? dequantizationFullPath.data : dequantizationFullPath.convert;

        inputs[emptyPathIndex] = dequantizationEmptyPath.convert == nullptr ?
            ((dequantizationEmptyPath.data->get_output_element_type(0) == newMultiplyFullPathValues->get_output_element_type(0)) ?
                dequantizationEmptyPath.data :
                std::make_shared<op::TypeRelaxed<opset1::Convert>>(
                    dequantizationEmptyPath.data, newMultiplyFullPathValues->get_output_element_type(0))) :
            dequantizationEmptyPath.convert;
        inputs[fullPathIndex] = std::make_shared<opset1::Multiply>(
            newSubtractFullPathValues == nullptr ?
                fullPathInput :
                std::make_shared<opset1::Subtract>(fullPathInput, newSubtractFullPathValues),
            newMultiplyFullPathValues);

        newMultiply = std::make_shared<opset1::Multiply>(
            std::make_shared<op::TypeRelaxed<opset1::Add>>(inputs[0], inputs[1]),
            multiplyEmptyPathValues);

        replace_node(add, newMultiply);
    }

    updateOutput(context, newMultiply, add);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
