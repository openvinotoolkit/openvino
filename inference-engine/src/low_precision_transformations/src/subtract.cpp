// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/subtract.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void SubtractTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Subtract>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Subtract>({ make_op_label<opset1::Convert>(), make_op_label<opset1::Constant>() }));
}

bool SubtractTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<opset1::Subtract>(m.get_match_root());
    if (!canBeTransformed(context, subtract)) {
        return false;
    }

    const ngraph::element::Type originalPrecision = subtract->get_output_element_type(0);

    const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(subtract);
    if (dequantization.multiply != nullptr) {
        // before: Y = X * SC - SH, after:  Y = (X - SH') * SC
        //    X * SC - SH = X * SC - SH' * SC
        //    SH' = SH / SC
        std::shared_ptr<opset1::Subtract> newSubtract = as_type_ptr<opset1::Subtract>(subtract->copy_with_new_inputs({
            dequantization.multiply->get_input_node_shared_ptr(0),
            ngraph::pass::low_precision::fold<ngraph::opset1::Divide>(
                subtract->get_input_node_shared_ptr(1),
                dequantization.multiply->get_input_node_shared_ptr(1))
        }));

        std::shared_ptr<Node> newMultiply = dequantization.multiply->copy_with_new_inputs({
            newSubtract,
            dequantization.multiply->input_value(1)
        });

        replace_node(subtract, newMultiply);
        subtract = newSubtract;
    }

    if (dequantization.subtract != nullptr) {
        std::shared_ptr<opset1::Subtract> newSubtract = as_type_ptr<opset1::Subtract>(subtract->copy_with_new_inputs({
            dequantization.subtract->get_input_node_shared_ptr(0),
            ngraph::pass::low_precision::fold<ngraph::opset1::Add>(
                subtract->get_input_node_shared_ptr(1),
                dequantization.subtract->get_input_node_shared_ptr(1))
        }));

        replace_node(subtract, newSubtract);
        subtract = newSubtract;
    }

    if (dequantization.convert != nullptr) {
        std::shared_ptr<Node> newSubtract = NetworkHelper::optimizeSubtract(subtract);
        newSubtract->set_output_type(0, originalPrecision, newSubtract->get_output_partial_shape(0));

        replace_node(newSubtract, std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
            newSubtract->get_input_node_shared_ptr(0),
            newSubtract->get_input_node_shared_ptr(1)));
    }
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
