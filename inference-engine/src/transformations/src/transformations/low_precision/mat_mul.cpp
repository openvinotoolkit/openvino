// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mat_mul.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "ngraph_ops/multiply_add.hpp"
#include "transformations/low_precision/network_helper.hpp"

using namespace ngraph;
using namespace ngraph::pass;
using namespace ngraph::pass::low_precision;

// TODO: not completed: not all dimensions are covered
void MatMulTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    const std::shared_ptr<ngraph::Node> matMul = m.get_match_root();
    if (!canBeTransformed(context, matMul)) {
        return;
    }

    const std::shared_ptr<ngraph::Node> multiply1 = matMul->input_value(0).get_node_shared_ptr();
    const std::shared_ptr<ngraph::Node> parent1 = multiply1->get_input_node_shared_ptr(0);
    const std::shared_ptr<ngraph::Node> multiply1Values = multiply1->get_input_node_shared_ptr(1);

    const std::shared_ptr<ngraph::Node> multiply2 = matMul->input_value(1).get_node_shared_ptr();
    const std::shared_ptr<ngraph::Node> parent2 = multiply2->get_input_node_shared_ptr(0);
    const std::shared_ptr<ngraph::Node> multiply2Values = multiply2->get_input_node_shared_ptr(1);

    // TODO: question (?): why we need 3 different methods if type values should be equal (before type relax)?
    // layer->get_element_type();
    // layer->get_output_element_type(i);
    // layer->get_input_element_type(i);
    //const ngraph::element::Type precision = matMul->get_element_type();

    const std::shared_ptr<ngraph::Node> resultMultiplyValues = fold<ngraph::opset1::Multiply>(multiply1Values, multiply2Values);

    // TODO: do I need to add any Convert layer routines
    const std::shared_ptr<opset1::Multiply> newMultiply = std::make_shared<opset1::Multiply>(
        std::make_shared<opset1::MatMul>(parent1, parent2),
        resultMultiplyValues);
    replace_node(matMul, newMultiply);

    // TODO: potentially name is not equal
    newMultiply->set_friendly_name(matMul->get_friendly_name());
}

void MatMulTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::MatMul>({ make_op_label<ngraph::opset1::Multiply>(), make_op_label<ngraph::opset1::Multiply>() }));
}

//bool MatMulTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
//    return true;
//}

bool MatMulTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    return true;
}
