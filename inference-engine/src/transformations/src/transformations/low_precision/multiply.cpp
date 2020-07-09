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
    std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(m.get_match_root());
    // const std::string multiplyName = multiply->get_friendly_name();
    // const ngraph::element::Type originalPrecision = multiply->get_output_element_type(0);
    // std::shared_ptr<Node> lastOperation = multiply;

//     const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(multiply);
//     if (dequantization.multiply != nullptr) {
//         // TODO: NO TESTS!!!
//         // before: Y = (X - SH) * SC1 * SC2, after:  Y = X * SC1 * SC2 - SH'
//         //    X * SC1 * SC2 - SH * SC1 * SC2 = X * SC1 * SC2 - SH'
//         //    SH' = SH * SC1 * SC2
//         std::shared_ptr<opset1::Multiply> newMultiply = as_type_ptr<opset1::Multiply>(multiply->copy_with_new_inputs({
//             dequantization.multiply->get_input_node_shared_ptr(0),
//             ngraph::pass::low_precision::fold<ngraph::opset1::Multiply>(
//                 // SC1
//                 dequantization.multiply->get_input_node_shared_ptr(1),
//                 // SC2
//                 multiply->get_input_node_shared_ptr(1))
//         }));

//         replace_node(multiply, newMultiply);
//         multiply = newMultiply;

    const int fullPathIndex = getNotEmpty(multiply);
    if (fullPathIndex == -1) {
        return;
    }

    const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

    FakeQuantizeDequantization dequantizationEmptyPath = getDequantization(multiply, emptyPathIndex);
    if (dequantizationEmptyPath.multiply == nullptr && dequantizationEmptyPath.subtract == nullptr) {
        return;
    }

    auto const dequantizationValuesEmptyPath = createEmptyValues(dequantizationEmptyPath);
    std::shared_ptr<Node> subtractValuesEmptyPath = std::get<0>(dequantizationValuesEmptyPath);
    std::shared_ptr<Node> multiplyValuesEmptyPath = std::get<1>(dequantizationValuesEmptyPath);

    // check if empty path shifts are not zero
    std::shared_ptr<opset1::Constant> subtract1ValuesConst = as_type_ptr<opset1::Constant>(subtractValuesEmptyPath);
    if (isScalarLike(subtract1ValuesConst)) {
        auto scalar = distillToScalar(subtract1ValuesConst);
        if (!op::util::constantIsEqualTo(scalar, 0)) {
            return;
        }
    }

    FakeQuantizeDequantization dequantizationFullPath = getDequantization(multiply, fullPathIndex);
    auto const dequantizationValuesFullpath = createEmptyValues(dequantizationFullPath);
    std::shared_ptr<Node> subtractValuesFullPath = std::get<0>(dequantizationValuesFullpath);
    std::shared_ptr<Node> multiplyValuesFullPath = std::get<1>(dequantizationValuesFullpath);

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

    std::shared_ptr<Node> newMultiply = std::make_shared<opset1::Multiply>(inputs[0], inputs[1]);

    replace_node(multiply, newMultiply);

    // TODO: NAMES!
    lastOperation->set_friendly_name(multiplyName);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
