// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/transpose.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void TransposeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Transpose>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
}

void transposeDequantizationConstant(std::shared_ptr<Node>& transpose) {
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(transpose, 0);

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtract->get_input_node_ptr(1)->get_output_shape(0);
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0);
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return;
    }

    if (dequantization.multiply->get_input_node_ptr(1)->get_output_shape(0).size() > 1ul) {
        if (dequantization.subtract != nullptr) {
            replace_node(
                dequantization.subtract->get_input_node_shared_ptr(1),
                fold<opset1::Transpose>(dequantization.subtract->get_input_node_shared_ptr(1), transpose->get_input_node_shared_ptr(1)));
        }

        if (dequantization.multiply != nullptr) {
            replace_node(
                dequantization.multiply->get_input_node_shared_ptr(1),
                fold<opset1::Transpose>(dequantization.multiply->get_input_node_shared_ptr(1), transpose->get_input_node_shared_ptr(1)));
        }
    }
}

void TransposeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<Node> transpose = m.get_match_root();
    if (!canBeTransformed(context, transpose)) {
        removeConvertIfPossible(transpose);
        return;
    }

    transpose = separateInStandaloneBranch(transpose);
    transposeDequantizationConstant(transpose);
    moveDequantizationAfter(context, transpose, NetworkHelper::getDequantization(transpose, 0), false);
}

bool TransposeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool TransposeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
