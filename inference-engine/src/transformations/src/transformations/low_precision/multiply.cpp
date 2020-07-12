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
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Multiply>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));

    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Multiply>({ make_op_label<opset1::Convert>(), make_op_label<opset1::Constant>() }));
}

void MultiplyTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(m.get_match_root());
    const std::string multiplyName = multiply->get_friendly_name();
    const ngraph::element::Type originalPrecision = multiply->get_output_element_type(0);
    std::shared_ptr<Node> lastOperation = multiply;

    const FakeQuantizeDequantization dequantization = ngraph::pass::low_precision::NetworkHelper::getDequantization(multiply);
    if (dequantization.multiply != nullptr) {
        // TODO: NO TESTS!!!
        // before: Y = (X - SH) * SC1 * SC2, after:  Y = X * SC1 * SC2 - SH'
        //    X * SC1 * SC2 - SH * SC1 * SC2 = X * SC1 * SC2 - SH'
        //    SH' = SH * SC1 * SC2
        std::shared_ptr<opset1::Multiply> newMultiply = as_type_ptr<opset1::Multiply>(multiply->copy_with_new_inputs({
            dequantization.multiply->get_input_node_shared_ptr(0),
            ngraph::pass::low_precision::fold<ngraph::opset1::Multiply>(
                // SC1
                dequantization.multiply->get_input_node_shared_ptr(1),
                // SC2
                multiply->get_input_node_shared_ptr(1))
        }));

        replace_node(multiply, newMultiply);
        multiply = newMultiply;
    }

    if (dequantization.subtract != nullptr) {
        // TODO: NO TESTS!!!
        // remove Subtract
        std::shared_ptr<opset1::Subtract> newSubtract = as_type_ptr<opset1::Subtract>(dequantization.subtract->copy_with_new_inputs({
            multiply,
            ngraph::pass::low_precision::fold<ngraph::opset1::Multiply>(
                // SC1 * SC2
                multiply->get_input_node_shared_ptr(1),
                // SH
                dequantization.subtract->get_input_node_shared_ptr(1))
        }));

        replace_node(multiply, newSubtract);
        lastOperation = newSubtract;
    }

    if (dequantization.convert != nullptr) {
        // TODO: error for inputs: I8 & F32
        // std::shared_ptr<Node> newMultiply = std::make_shared<op::TypeRelaxed<opset1::Multiply>>(
        //    dequantization.convert->get_input_node_shared_ptr(0),
        //    multiply->get_input_node_shared_ptr(1));
        // replace_node(multiply, newMultiply);
        // newMultiply->set_output_type(0, originalPrecision, newMultiply->get_output_partial_shape(0));

        // TODO: workaround has to be removed: ConvertTransformation is used

        // TODO: workaround: replace Convert to Sub with zero
        const ngraph::element::Type convertOriginalPrecision = dequantization.convert->get_input_element_type(0);

        std::shared_ptr<opset1::Subtract> newConvert = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
            dequantization.convert->get_input_node_shared_ptr(0),
            std::make_shared<opset1::Constant>(convertOriginalPrecision, Shape{}, std::vector<size_t>({ 0 })));
        newConvert->set_output_type(0, originalPrecision, dequantization.convert->get_output_partial_shape(0));

        std::shared_ptr<Node> newMultiply = std::make_shared<opset1::Multiply>(
            newConvert,
            multiply->get_input_node_shared_ptr(1));
        replace_node(multiply, newMultiply);

        if (dequantization.subtract == nullptr) {
            lastOperation = newMultiply;
        }
    }

    // TODO: NAMES!
    lastOperation->set_friendly_name(multiplyName);

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
