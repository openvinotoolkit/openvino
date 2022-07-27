// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/strided_slice.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>
#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

namespace {

std::shared_ptr<opset1::Constant> stridedSliceDeqConstant(
    const std::shared_ptr<ngraph::Node> strSlice,
    const std::shared_ptr<ngraph::Node> dequantizaitonConstant) {
    auto constant = ov::as_type_ptr<ngraph::opset1::Constant>(dequantizaitonConstant);
    auto constantShape = constant->get_shape();
    if (shape_size(constantShape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    const auto stridedSlicePShape = strSlice->get_input_partial_shape(0);
    const size_t rank = stridedSlicePShape.rank().get_length();
    if (rank != constantShape.size()) {
        ngraph::Shape newConstantShape;
        if (ngraph::shape_size(constantShape) == 1) {
            newConstantShape = ngraph::Shape(rank, 1);
        } else {
            newConstantShape = constantShape;

            // case when constShape without batch
            if ((constantShape.size() > 1) &&
                (constantShape.size() < rank)) {
                newConstantShape.insert(newConstantShape.begin(), 1);
            }
        }
        constantShape = newConstantShape;

        const auto newConstant = fold<ngraph::opset1::Broadcast>(
            constant,
            ngraph::opset1::Constant::create(ngraph::element::i32, { newConstantShape.size() }, newConstantShape));
        constant = ov::as_type_ptr<ngraph::opset1::Constant>(newConstant);
    }

    const auto stridedSlice = ov::as_type_ptr<ngraph::opset1::StridedSlice>(strSlice);

    auto beginMask = stridedSlice->get_begin_mask();
    auto endMask = stridedSlice->get_end_mask();
    for (size_t i = 0; i < constantShape.size(); ++i) {
        if ((beginMask.size() <= i) && (endMask.size() <= i)) {
            break;
        }
        // don't slice constant if current dimension is 1
        if (constantShape[i] == 1ul) {
            beginMask[i] = 1ul;
            endMask[i] = 1ul;
        }
    }

    const auto result = fold<ngraph::opset1::StridedSlice>(
        constant,
        stridedSlice->input_value(1),
        stridedSlice->input_value(2),
        stridedSlice->input_value(3),
        beginMask,
        endMask,
        stridedSlice->get_new_axis_mask(),
        stridedSlice->get_shrink_axis_mask(),
        stridedSlice->get_ellipsis_mask());

    return ov::as_type_ptr<opset1::Constant>(NetworkHelper::toScalarIfPossible(result));
}

} // namespace

StridedSliceTransformation::StridedSliceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(StridedSliceTransformation);
    auto matcher = ngraph::pattern::wrap_type<opset1::StridedSlice>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool StridedSliceTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    if (!StridedSliceTransformation::canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto stridedSlice = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    auto dequantization = NetworkHelper::getDequantization(stridedSlice, defaultPrecisions);

    if (dequantization.subtract) {
        const auto newSubConst = stridedSliceDeqConstant(stridedSlice, dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, newSubConst);
        dequantization.subtractConstant = newSubConst;
    }

    const auto newMulConst = stridedSliceDeqConstant(stridedSlice, dequantization.multiplyConstant);
    replace_node(dequantization.multiplyConstant, newMulConst);
    dequantization.multiplyConstant = newMulConst;

    moveDequantizationAfter(context, stridedSlice, NetworkHelper::getDequantization(stridedSlice, defaultPrecisions), false);
    return true;
}

bool StridedSliceTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> operation) const {
    if (!ov::is_type<ngraph::opset1::StridedSlice>(operation)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(operation);
    if (dequantization.empty()) {
        return false;
    }

    if (operation->get_input_partial_shape(0).rank().is_dynamic() &&
        ((dequantization.subtract && ngraph::shape_size(dequantization.subtractConstant->get_shape()) > 1) ||
         (dequantization.multiply && ngraph::shape_size(dequantization.multiplyConstant->get_shape()) > 1))) {
        return false;
    }

    return true;
}

bool StridedSliceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
