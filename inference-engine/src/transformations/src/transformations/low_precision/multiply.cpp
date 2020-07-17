// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/multiply.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void MultiplyTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addSingleNodePattern<opset1::Multiply>(pass, context);
}

void MultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto multiply = m.get_match_root();

    multiply = separateInStandaloneBranch(multiply);
    auto newMultiply = multiply;

    const int fullPathIndex = getNotEmpty(multiply);

    if (fullPathIndex == -1) {
        const auto multiplyBranch = getMultiplyConstBranch(multiply);

        if (multiplyBranch.first == -1 || multiplyBranch.second == -1)
            return;

        auto multiplyParent = multiply->get_input_node_shared_ptr(multiplyBranch.first);
        auto constParent = multiply->get_input_node_shared_ptr(multiplyBranch.first == 0 ? 1 : 0);
        auto multiplyParentParent = multiplyParent->get_input_node_shared_ptr(multiplyBranch.second);
        auto multiplyParentConst = multiplyParent->get_input_node_shared_ptr(multiplyBranch.second == 0 ? 1 : 0);

        newMultiply = std::make_shared<opset1::Multiply>(
            multiplyParentParent,
            fold<opset1::Multiply>(multiplyParentConst, constParent));
    } else {
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        FakeQuantizeDequantization dequantizationEmptyPath = NetworkHelper::getDequantization(multiply, emptyPathIndex);
        if (dequantizationEmptyPath.multiply == nullptr && dequantizationEmptyPath.subtract == nullptr) {
            return;
        }

        std::shared_ptr<Node> subtractValuesEmptyPath;
        std::shared_ptr<Node> multiplyValuesEmptyPath;
        std::tie(subtractValuesEmptyPath, multiplyValuesEmptyPath) = NetworkHelper::createEmptyValues(dequantizationEmptyPath);

        // check if empty path shifts are not zero
        if (!NetworkHelper::isZeroConst(subtractValuesEmptyPath)) {
            return;
        }

        FakeQuantizeDequantization dequantizationFullPath = NetworkHelper::getDequantization(multiply, fullPathIndex);
        std::shared_ptr<Node> subtractValuesFullPath;
        std::shared_ptr<Node> multiplyValuesFullPath;
        std::tie(subtractValuesFullPath, multiplyValuesFullPath) = NetworkHelper::createEmptyValues(dequantizationFullPath);


        // before: Y = (SC1 * (X1 - SH1)) * (SC2 * X2)
        // after : Y = (SC1' * (X1 - SH1)) * (X2) , where :
        //         SC1' = SC1 * SC2
        std::shared_ptr<Node> newMultiplyValuesFullPath = fold<opset1::Multiply>(multiplyValuesEmptyPath, multiplyValuesFullPath);
        std::vector<std::shared_ptr<Node>> inputs{ {}, {} };
        inputs[emptyPathIndex] = dequantizationEmptyPath.subtract == nullptr ?
            dequantizationEmptyPath.multiply->get_input_node_shared_ptr(0) :
            dequantizationEmptyPath.subtract->get_input_node_shared_ptr(0);
        inputs[fullPathIndex] = std::make_shared<opset1::Multiply>(
            dequantizationFullPath.subtract == nullptr ?
                (dequantizationFullPath.convert == nullptr ?
                    dequantizationFullPath.data : dequantizationFullPath.convert) :
                dequantizationFullPath.subtract,
            newMultiplyValuesFullPath);

        newMultiply = std::make_shared<opset1::Multiply>(inputs[0], inputs[1]);
    }

    replace_node(multiply, newMultiply);
    updateOutput(context, newMultiply, multiply);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
