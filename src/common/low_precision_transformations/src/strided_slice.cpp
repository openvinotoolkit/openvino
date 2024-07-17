// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/network_helper.hpp"
#include "low_precision/strided_slice.hpp"

namespace ov {
namespace pass {
namespace low_precision {

namespace {

std::shared_ptr<ov::opset1::Constant> stridedSliceDeqConstant(
    const std::shared_ptr<Node> node,
    const std::shared_ptr<Node> dequantizaiton_constant) {
    const auto constant = ov::as_type_ptr<ov::opset1::Constant>(dequantizaiton_constant);
    const auto& original_constant_shape = constant->get_shape();
    if (shape_size(original_constant_shape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    }

    // step #1: align shapes
    std::shared_ptr<ov::opset1::Constant> new_constant = constant;
    const size_t rank = node->get_input_partial_shape(0).size();
    Shape new_constant_shape = original_constant_shape;
    if (rank != new_constant_shape.size()) {
        // case when constant shape without batch
        if (original_constant_shape.size() < rank) {
            new_constant_shape.insert(new_constant_shape.begin(), 1);
        }

        if (original_constant_shape != new_constant_shape) {
            const auto result = fold<ov::opset1::Broadcast>(
                constant,
                ov::opset1::Constant::create(ov::element::i32, { new_constant_shape.size() }, new_constant_shape));
            new_constant = ov::as_type_ptr<ov::opset1::Constant>(result);
        }
    }

    // step #2: update original begin & end & strides
    auto cast_vector = [](const std::shared_ptr<ov::opset1::StridedSlice>& strided_slice, const size_t i) {
        const auto constant = ov::util::get_constant_from_source(strided_slice->get_input_source_output(i));
        assert(constant != nullptr);
        return constant->cast_vector<int64_t>();
    };

    const auto strided_slice = ov::as_type_ptr<ov::opset1::StridedSlice>(node);
    auto begin = cast_vector(strided_slice, 1);
    auto end = cast_vector(strided_slice, 2);
    auto strides = cast_vector(strided_slice, 3);
    auto begin_mask = strided_slice->get_begin_mask();
    auto end_mask = strided_slice->get_end_mask();
    for (auto i = 0ull; i < new_constant_shape.size(); ++i) {
        // don't slice constant if current dimension is 1
        if (new_constant_shape[i] == 1ull) {
            if (i < begin.size()) {
                begin[i] = 0;
            }
            if (i < end.size()) {
                end[i] = 0;
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

    return ov::as_type_ptr<ov::opset1::Constant>(NetworkHelper::toScalarIfPossible(result));
}

} // namespace

StridedSliceTransformation::StridedSliceTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(StridedSliceTransformation);
    auto matcher = ov::pass::pattern::wrap_type<ov::opset1::StridedSlice>();

    ov::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool StridedSliceTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    if (!StridedSliceTransformation::canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto strided_slice = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    auto dequantization = NetworkHelper::getDequantization(strided_slice, defaultPrecisions);

    if (dequantization.subtract) {
        const auto new_sub_const = stridedSliceDeqConstant(strided_slice, dequantization.subtractConstant);
        replace_node(dequantization.subtractConstant, new_sub_const);
        dequantization.subtractConstant = new_sub_const;
    }

    const auto new_mul_const = stridedSliceDeqConstant(strided_slice, dequantization.multiplyConstant);
    replace_node(dequantization.multiplyConstant, new_mul_const);
    dequantization.multiplyConstant = new_mul_const;

    const auto newOperation = moveDequantizationAfter(context, strided_slice, NetworkHelper::getDequantization(strided_slice, defaultPrecisions));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
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

    const auto is_dequantization_scalar =
        ((dequantization.subtract && shape_size(dequantization.subtractConstant->get_shape()) == 1ull) &&
        (dequantization.multiply && shape_size(dequantization.multiplyConstant->get_shape()) == 1ull));

    if (operation->get_input_partial_shape(0).rank().is_dynamic() && !is_dequantization_scalar) {
        return false;
    }

    return is_dequantization_scalar || (ov::util::get_constant_from_source(operation->get_input_source_output(1)) &&
                                        ov::util::get_constant_from_source(operation->get_input_source_output(2)) &&
                                        ov::util::get_constant_from_source(operation->get_input_source_output(3)));
}

bool StridedSliceTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}
} // namespace low_precision
} // namespace pass
} // namespace ov
