// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/strided_slice.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

std::shared_ptr<Node> stridedSliceDeqConstant(
    const std::shared_ptr<ngraph::Node> strSlice,
    const std::shared_ptr<ngraph::Node> dequantizaitonConstant) {
    auto constant = as_type_ptr<ngraph::opset1::Constant>(dequantizaitonConstant);
    // issue #48857: constant is mistakenly recognized as a scalar. Uncomment after fix
    //if (NetworkHelper::isScalarLike(constant)) {
    //    return NetworkHelper::toScalar(constant);
    //}

    if (strSlice->get_input_shape(0).size() != constant->get_shape().size()) {
        const auto constantShape = constant->get_shape();
        const auto stridedSliceShape = strSlice->get_input_shape(0);
        ngraph::Shape newConstantShape(stridedSliceShape.size(), 1);

        for (size_t i = 0; i < constantShape.size(); ++i) {
            if (constantShape[i] != 1) {
                newConstantShape[i] = constantShape[i];
            }
        }

        const auto newConstant = fold<ngraph::opset1::Broadcast>(
            constant,
            ngraph::opset1::Constant::create(ngraph::element::i32, { newConstantShape.size() }, newConstantShape));
        constant = as_type_ptr<ngraph::opset1::Constant>(newConstant);
    }

    const auto stridedSlice = as_type_ptr<ngraph::opset1::StridedSlice>(strSlice);
    const auto result = fold<ngraph::opset1::StridedSlice>(
        constant,
        stridedSlice->get_input_node_shared_ptr(1),
        stridedSlice->get_input_node_shared_ptr(2),
        stridedSlice->get_input_node_shared_ptr(3),
        stridedSlice->get_begin_mask(),
        stridedSlice->get_end_mask(),
        stridedSlice->get_new_axis_mask(),
        stridedSlice->get_shrink_axis_mask(),
        stridedSlice->get_ellipsis_mask());

    return NetworkHelper::toScalarIfPossible(result);
}

StridedSliceTransformation::StridedSliceTransformation(const Params& params) : LayerTransformation(params) {}

void StridedSliceTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
        context,
        make_op_pattern<opset1::StridedSlice>({
            make_op_label<opset1::Multiply>(),
            make_op_label<opset1::Constant>(),
            make_op_label<opset1::Constant>(),
            make_op_label<opset1::Constant>() }));
}

bool StridedSliceTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!StridedSliceTransformation::canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto stridedSlice = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    const auto dequantization = NetworkHelper::getDequantization(stridedSlice);

    if (dequantization.subtract) {
        const auto subConst = NetworkHelper::getConstantInput(dequantization.subtract);
        const size_t subConstIdx = NetworkHelper::getChildInputIndex(subConst, dequantization.subtract);

        const auto newSubConst = stridedSliceDeqConstant(stridedSlice, subConst);
        dequantization.subtract->set_argument(subConstIdx, newSubConst);
    }

    const auto mulConst = NetworkHelper::getConstantInput(dequantization.multiply);
    const size_t mulConstIdx = NetworkHelper::getChildInputIndex(mulConst, dequantization.multiply);

    const auto newMulConst = stridedSliceDeqConstant(stridedSlice, mulConst);
    dequantization.multiply->set_argument(mulConstIdx, newMulConst);

    moveDequantizationAfter(context, stridedSlice, NetworkHelper::getDequantization(stridedSlice), false);
    return true;
}

bool StridedSliceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!is_type<ngraph::opset1::StridedSlice>(operation)) {
        return false;
    }

    return !NetworkHelper::getDequantization(operation).empty();
}

bool StridedSliceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
