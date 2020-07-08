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

#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/type_relaxed.hpp"

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

using namespace ngraph;

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

typedef std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> FakeQuantizeDequantizationValues;

FakeQuantizeDequantizationValues createEmptyValues(const FakeQuantizeDequantization& dequantization) {
    std::shared_ptr<Node> parent = dequantization.convert ? dequantization.convert : dequantization.data;

    std::shared_ptr<Node> multiply1Const = dequantization.multiply ?
        dequantization.multiply->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 1.f }));

    std::shared_ptr<Node> subtract1Const = dequantization.subtract ?
        dequantization.subtract->get_input_node_shared_ptr(1)->clone_with_new_inputs({}) :
        std::make_shared<opset1::Constant>(parent->get_output_element_type(0), Shape({}), std::vector<float>({ 0.f }));

    subtract1Const->set_output_type(0, multiply1Const->get_output_element_type(0), subtract1Const->get_output_partial_shape(0));

    return FakeQuantizeDequantizationValues(subtract1Const, multiply1Const);
}

void AddTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // TODO: move to handler
    if (!canBeTransformed(context, m.get_match_root())) {
        return;
    }

    const std::shared_ptr<ngraph::opset1::Add> add = as_type_ptr<opset1::Add>(m.get_match_root());

    // pass::VisualizeTree("C:\\Projects\\temp\\test.original").run_on_module(std::vector<std::shared_ptr<Function>>{ context.network });

    // Figure out where SS and where is Constant
    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(add->input_value(0).get_node_shared_ptr());
    std::shared_ptr<opset1::Multiply> multiply;
    if (constant) {
        multiply = as_type_ptr<opset1::Multiply>(add->input_value(1).get_node_shared_ptr());
    } else {
        constant = as_type_ptr<opset1::Constant>(add->input_value(1).get_node_shared_ptr());
        multiply = as_type_ptr<opset1::Multiply>(add->input_value(0).get_node_shared_ptr());
    }

    std::shared_ptr<opset1::Multiply> newMultiply;
    if ((constant != nullptr) && (multiply != nullptr)) {
        newMultiply = swapMultiplyAndAdd(add);
    } else {
        const int fullPathIndex = getNotEmpty(add);
        if (fullPathIndex == -1) {
            return;
        }
        const int emptyPathIndex = fullPathIndex == 0 ? 1 : 0;

        // TODO: question: is it reasonable to create Constant? (performance issue?)
        // TODO: question: should we clone constant here?

        FakeQuantizeDequantization dequantization1 = pass::low_precision::getDequantization(add, emptyPathIndex);
        auto const dequantizationValues1 = createEmptyValues(dequantization1);
        std::shared_ptr<Node> subtract1Values = std::get<0>(dequantizationValues1);
        std::shared_ptr<Node> multiply1Values = std::get<1>(dequantizationValues1);

        FakeQuantizeDequantization dequantization2 = pass::low_precision::getDequantization(add, fullPathIndex);
        auto const dequantizationValues2 = createEmptyValues(dequantization2);
        std::shared_ptr<Node> subtract2Values = std::get<0>(dequantizationValues2);
        std::shared_ptr<Node> multiply2Values = std::get<1>(dequantizationValues2);

        // calculation
        std::shared_ptr<Node> newSubtract2Values = fold<opset1::Add>(
            subtract2Values,
            fold<opset1::Divide>(
                fold<opset1::Multiply>(subtract1Values, multiply1Values),
                multiply2Values));

        std::shared_ptr<Node> newMultiply2Const = fold<opset1::Divide>(multiply2Values, multiply1Values);

        // result optimization
        {
            // empty Subtract after calculations removing
            std::shared_ptr<opset1::Constant> newSubtract2ConstOp = as_type_ptr<opset1::Constant>(newSubtract2Values);
            if ((newSubtract2ConstOp != nullptr) && isScalarLike(newSubtract2ConstOp)) {
                auto scalar = distillToScalar(newSubtract2ConstOp);
                if (op::util::constantIsEqualTo(scalar, 0)) {
                    newSubtract2Values = nullptr;
                }
            }
        }

        // graph update
        newMultiply = std::make_shared<opset1::Multiply>(
            std::make_shared<op::TypeRelaxed<opset1::Add>>(
                dequantization1.convert == nullptr ?
                    ((dequantization1.data->get_output_element_type(0) == newMultiply2Const->get_output_element_type(0)) ?
                        dequantization1.data :
                        std::make_shared<op::TypeRelaxed<opset1::Convert>>(dequantization1.data, newMultiply2Const->get_output_element_type(0))) :
                    dequantization1.convert,
                std::make_shared<opset1::Multiply>(
                    newSubtract2Values == nullptr ?
                        dequantization2.convert == nullptr ? dequantization2.data : dequantization2.convert :
                        std::make_shared<opset1::Subtract>(dequantization2.convert, newSubtract2Values),
                    newMultiply2Const)
            ),
            multiply1Values
        );

        replace_node(add, newMultiply);
    }

    // TODO: FIXME: output names
    newMultiply->set_friendly_name(add->get_friendly_name());

    // std::cout << "AddTransformation::transform: done: " << newMultiply->get_friendly_name() << std::endl;

    // pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<Function>>{ context.network });
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
