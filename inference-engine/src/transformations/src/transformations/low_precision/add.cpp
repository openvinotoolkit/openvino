// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "ngraph_ops/type_relaxed.hpp"

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void AddTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Add>(pass, context);
}

bool AddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::Add> op = as_type_ptr<opset1::Add>(m.get_match_root());
    if (!canBeTransformed(context, op)) {
        return false;
    }

    std::shared_ptr<opset1::Add> add = as_type_ptr<opset1::Add>(separateInStandaloneBranch(op));
    const int fullPathIndex = getNotEmpty(add);
    std::shared_ptr<Node> newMultiply;
    std::shared_ptr<Node> newAdd;

    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(add);

        if (multiplyBranch.first == -1)
            return false;

        newMultiply = NetworkHelper::swapMultiplyAndAdd(add, multiplyBranch.first);
        if (is_type<opset1::Add>(newMultiply->get_input_node_shared_ptr(multiplyBranch.first))) {
            newAdd = as_type_ptr<opset1::Add>(newMultiply->get_input_node_shared_ptr(0));
        } else {
            newAdd = newMultiply;
        }
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

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

        inputs[emptyPathIndex] = dequantizationEmptyPath.data;
        inputs[fullPathIndex] = std::make_shared<opset1::Multiply>(
            newSubtractFullPathValues == nullptr ?
                fullPathInput :
                std::make_shared<opset1::Subtract>(fullPathInput, newSubtractFullPathValues),
            newMultiplyFullPathValues);

        newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(
            std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{ element::f32 },
            ngraph::op::TemporaryReplaceOutputType(inputs[0], element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(inputs[1], element::f32).get());
        newMultiply = std::make_shared<opset1::Multiply>(newAdd, multiplyEmptyPathValues);

        replace_node(add, newMultiply);
    }

    NetworkHelper::copyInfo(add, newAdd);
    updateOutput(context, newMultiply, newAdd);
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
