// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/max_pool.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void MaxPoolTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MaxPool>({ make_op_label<opset1::Multiply>() }));
}

void MaxPoolTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    // VisualizeTree("C:\\Projects\\temp\\test.original.max_pool").run_on_module(std::vector<std::shared_ptr<Function>>{ context.network });

    std::shared_ptr<Node> pooling = m.get_match_root();

    // TODO: dequantization operation handling: getDequantizationOperations()
    // TODO: any operation below can have several children - should be handled in getDequantizationOperations()

    std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(pooling->input_value(0).get_node_shared_ptr());
    if (multiply == nullptr) {
        THROW_IE_LPT_EXCEPTION(*pooling) << "not expected dequantization operation type";
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

    std::shared_ptr<Node> newPooling = pooling->copy_with_new_inputs({ dataNode });

    std::shared_ptr<Node> replacement = multiply->copy_with_new_inputs({
        subtract ?
            (convert ?
                subtract->copy_with_new_inputs({convert->copy_with_new_inputs({newPooling}), subtract->get_input_node_shared_ptr(1)}) :
                subtract->copy_with_new_inputs({newPooling, subtract->get_input_node_shared_ptr(1)})) :
            (convert ? convert->copy_with_new_inputs({newPooling}) : newPooling),
        multiply->input_value(1) });

    NetworkHelper::setOutDataPrecision(newPooling, newPooling->get_input_element_type(0));
    replace_node(pooling, replacement);

    // VisualizeTree("C:\\Projects\\temp\\test.transformed.max_pool").run_on_module(std::vector<std::shared_ptr<Function>>{ context.network });

    // TODO: NAMES!
}

bool MaxPoolTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
