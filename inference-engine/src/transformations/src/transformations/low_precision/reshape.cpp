// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/reshape.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void ReshapeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Reshape>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void ReshapeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> reshape = m.get_match_root();

    // TODO: dequantization operation handling: getDequantizationOperations()
    // TODO: any operation below can have several children - should be handled in getDequantizationOperations()

    std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(reshape->input_value(0).get_node_shared_ptr());
    if (multiply == nullptr) {
        THROW_IE_LPT_EXCEPTION(*reshape) << "not expected dequantization operation type";
    }

    // TODO: move to canBeTransformed?
    std::shared_ptr<opset1::Constant> scaleConstant = as_type_ptr<opset1::Constant>(multiply->input_value(1).get_node_shared_ptr());
    const std::vector<float> scales = scaleConstant->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](float value) { return value < 0.0; })) {
        return;
    }

    std::shared_ptr<Node> parent = multiply->input_value(0).get_node_shared_ptr();
    std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<opset1::Subtract>(parent);
    if (subtract != nullptr) {
        parent = subtract->get_input_node_shared_ptr(0);
    }

    std::shared_ptr<opset1::Convert> convert = as_type_ptr<opset1::Convert>(parent);
    if (convert != nullptr) {
        parent = convert->get_input_node_shared_ptr(0);
    }

    const Output<Node> dataNode = parent;

    // TODO: dequantization operation handling: insertDequantizationOperations()

    // std::shared_ptr<Node> newReshape = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Reshape>>(
    //    dataNode, reshape->input_value(1).get_node_shared_ptr());
    std::shared_ptr<Node> newReshape = reshape->copy_with_new_inputs({ dataNode, reshape->input_value(1).get_node_shared_ptr() });


    // TODO: Multiply Const has to change shape
    std::shared_ptr<Node> replacement = multiply->copy_with_new_inputs({
        subtract ?
            (convert ?
                subtract->copy_with_new_inputs({convert->copy_with_new_inputs({newReshape}), subtract->get_input_node_shared_ptr(1)}) :
                subtract->copy_with_new_inputs({newReshape, subtract->get_input_node_shared_ptr(1)})) :
            (convert ? convert->copy_with_new_inputs({newReshape}) : newReshape),
        multiply->input_value(1) });
    replace_node(reshape, replacement);

    // auto elementType = newReshape->get_input_element_type(0);
    // NetworkHelper::setOutDataPrecision(newReshape, elementType);

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ context.network });

    // TODO: NAMES!
    replacement->set_friendly_name(reshape->get_friendly_name());
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
