// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/transpose.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/util/log.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ov {
namespace pass {
namespace low_precision {

TransposeTransformation::TransposeTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(TransposeTransformation);
    auto matcher = pattern::wrap_type<ov::opset1::Transpose>({ pattern::wrap_type<ov::opset1::Multiply>(), pattern::wrap_type<ov::opset1::Constant>() });

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

namespace {

void transposeDequantizationConstant(std::shared_ptr<Node>& transpose, const std::vector<ov::element::Type>& defaultPrecisions) {
    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(transpose, defaultPrecisions);

    const Shape subtractShape = dequantization.subtract == nullptr ? Shape{} : dequantization.subtractConstant->get_shape();
    const Shape multiplyShape = dequantization.multiply == nullptr ? Shape{} : dequantization.multiplyConstant->get_shape();
    if ((subtractShape.empty() || (subtractShape.size() == 1ul)) && (multiplyShape.empty() || (multiplyShape.size() == 1ul))) {
        return;
    }

    auto transposeDeqConstant = [](
        const std::shared_ptr<ov::opset1::Constant>& dequantizationConstant,
        const PartialShape& transposeOutputPShape,
        const std::shared_ptr<Node>& transposeConstant) -> std::shared_ptr<Node> {
            const auto constantShape = dequantizationConstant->get_shape();
            if (shape_size(constantShape) == 1ul) {
                return NetworkHelper::toScalar(dequantizationConstant);
            }

            assert(transposeOutputPShape.rank().is_static());
            const size_t transposeOutRank = transposeOutputPShape.rank().get_length();
            if (constantShape.size() != transposeOutRank) {
                const auto unsqueezeConst = ov::opset1::Constant::create(element::i32, Shape{ 1 }, std::vector<size_t>{ 0 });
                const auto deqConstantWithBatch = fold<ov::opset1::Unsqueeze>(dequantizationConstant, unsqueezeConst);
                return fold<ov::opset1::Transpose>(deqConstantWithBatch, transposeConstant);
            } else {
                return fold<ov::opset1::Transpose>(dequantizationConstant, transposeConstant);
            }
    };

    if (dequantization.subtract != nullptr) {
        const auto constant = transposeDeqConstant(
            dequantization.subtractConstant,
            transpose->get_output_partial_shape(0),
            transpose->get_input_node_shared_ptr(1));
        replace_node(dequantization.subtractConstant, constant);
    }

    if (dequantization.multiply != nullptr) {
        const auto constant = transposeDeqConstant(
            dequantization.multiplyConstant,
            transpose->get_output_partial_shape(0),
            transpose->get_input_node_shared_ptr(1));
        replace_node(dequantization.multiplyConstant, constant);
    }
}

} // namespace

bool TransposeTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher &m) {
    std::shared_ptr<Node> transpose = m.get_match_root();
    if (!canBeTransformed(context, transpose)) {
        return false;
    }

    transpose = NetworkHelper::separateInStandaloneBranch(transpose, defaultPrecisions);
    transposeDequantizationConstant(transpose, defaultPrecisions);
    const auto newOperation = moveDequantizationAfter(context, transpose, NetworkHelper::getDequantization(transpose, defaultPrecisions, 0));

    OPENVINO_DEBUG("LPT: done: ", newOperation);
    return true;
}

bool TransposeTransformation::isPrecisionPreserved(std::shared_ptr<Node> op) const noexcept {
    return true;
}

bool TransposeTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformed(context, op)) {
        return false;
    }

    const std::shared_ptr<ov::opset1::Constant> constant = ov::as_type_ptr<ov::opset1::Constant>(op->get_input_node_shared_ptr(1));
    if (constant == nullptr) {
        return false;
    }

    const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(op, defaultPrecisions);
    const bool isPerTensor = [&] {
        if (dequantization.subtractConstant != nullptr) {
            if (!NetworkHelper::isScalarLike(dequantization.subtractConstant)) {
                return false;
            }
        }
        if (dequantization.multiply != nullptr) {
            const auto mulConst = ov::as_type_ptr<ov::op::v0::Constant>(dequantization.multiplyConstant);
            if (!NetworkHelper::isScalarLike(mulConst)) {
                return false;
            }
        }
        return true;
    }();

    // TODO: remove legacy limitation
    if (!isPerTensor) {
        const auto values = constant->cast_vector<float>();
        if ((values.size() < 2ul) || (values[0] != 0) || (values[1] != 1)) {
            return false;
        }
    }

    auto checkShape = [](const std::shared_ptr<ov::opset1::Constant>& dequantizationConstant, const PartialShape& transposeOutputShape) -> bool {
        const auto dequantizationShape = dequantizationConstant->get_shape();
        const auto rank = transposeOutputShape.rank();
        if (rank.is_dynamic()) {
            return false;
        }

        const size_t rankValue = rank.get_length();
        if (dequantizationShape.empty() || (dequantizationShape.size() == 1ul) || (dequantizationShape.size() == rankValue)) {
            return true;
        }

        if (dequantizationShape.size() > rankValue) {
            return false;
        }

        return (rankValue - dequantizationShape.size()) == 1;
    };

    return
        !dequantization.empty() &&
        ((dequantization.subtract == nullptr) || checkShape(dequantization.subtractConstant, op->get_output_partial_shape(0))) &&
        ((dequantization.multiply == nullptr) || checkShape(dequantization.multiplyConstant, op->get_output_partial_shape(0)));
}

} // namespace low_precision
} // namespace pass
} // namespace ov
