// Copyright (C) 2018-2023 Intel Corporation
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

std::shared_ptr<ov::opset1::Constant> stridedSliceDeqConstant(
    const std::shared_ptr<ngraph::Node> strSlice,
    const std::shared_ptr<ngraph::Node> dequantizaitonConstant) {
    const auto constant = ov::as_type_ptr<ov::opset1::Constant>(dequantizaitonConstant);
    const auto& original_constant_shape = constant->get_shape();
    if (shape_size(original_constant_shape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    // step #1: align shapes
    std::shared_ptr<ov::opset1::Constant> new_constant = constant;
    const size_t rank = strSlice->get_input_partial_shape(0).rank().get_length();
    ngraph::Shape newConstantShape = original_constant_shape;
    if (rank != newConstantShape.size()) {
        if (ngraph::shape_size(original_constant_shape) == 1) {
            newConstantShape = ngraph::Shape(rank, 1);
        } else {
            newConstantShape = original_constant_shape;

            // case when constShape without batch
            if ((original_constant_shape.size() > 1) &&
                (original_constant_shape.size() < rank)) {
                newConstantShape.insert(newConstantShape.begin(), 1);
            }
        }

        if (original_constant_shape != newConstantShape) {
            const auto newConstant = fold<ov::opset1::Broadcast>(
                constant,
                ov::opset1::Constant::create(ngraph::element::i32, { newConstantShape.size() }, newConstantShape));
            new_constant = ov::as_type_ptr<ov::opset1::Constant>(newConstant);
        }
    }

    // step #2: update original begin & end & strides
    const auto strided_slice = ov::as_type_ptr<ov::opset1::StridedSlice>(strSlice);
    auto begin = ov::as_type_ptr<ov::opset1::Constant>(strided_slice->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
    auto end = ov::as_type_ptr<ov::opset1::Constant>(strided_slice->get_input_node_shared_ptr(2))->cast_vector<int64_t>();
    auto strides = ov::as_type_ptr<ov::opset1::Constant>(strided_slice->get_input_node_shared_ptr(3))->cast_vector<int64_t>();
    auto begin_mask = strided_slice->get_begin_mask();
    auto end_mask = strided_slice->get_end_mask();
    for (auto i = 0ull; i < newConstantShape.size(); ++i) {
        // don't slice constant if current dimension is 1
        if (newConstantShape[i] == 1ull) {
            if (i < begin.size()) {
                begin[i] = 0;
            }
            if (i < end.size()) {
                end[i] = 1;
            }

            if (i < strides.size()) {
                strides[i] = 1;
            }

            if (i < begin_mask.size()) {
                begin_mask[i] = 1;
            }

            if (i < end_mask.size()) {
                end_mask[i] = 1;
            }
        }
    }

    // step #3: final step: dequantizatin constant folding
    const auto result = fold<ov::opset1::StridedSlice>(
        new_constant,
        std::make_shared<ov::opset1::Constant>(element::i64, Shape{ begin.size() }, begin),
        std::make_shared<ov::opset1::Constant>(element::i64, Shape{ end.size() }, end),
        std::make_shared<ov::opset1::Constant>(element::i64, Shape{ strides.size() }, strides),
        begin_mask,
        end_mask,
        strided_slice->get_new_axis_mask(),
        strided_slice->get_shrink_axis_mask(),
        strided_slice->get_ellipsis_mask());

    new_constant = ov::as_type_ptr<ov::opset1::Constant>(NetworkHelper::toScalarIfPossible(result));

    if (shape_size(new_constant->get_shape()) == 1ul) {
        return NetworkHelper::toScalar(new_constant);
    }

    return new_constant;
}

} // namespace

StridedSliceTransformation::StridedSliceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(StridedSliceTransformation);
    auto matcher = ngraph::pattern::wrap_type<ov::opset1::StridedSlice>();

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
    if (!ov::is_type<ov::opset1::StridedSlice>(operation)) {
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
