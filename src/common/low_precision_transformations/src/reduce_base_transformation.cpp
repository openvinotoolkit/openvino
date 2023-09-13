// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_base_transformation.hpp"
#include <memory>

#include "low_precision/network_helper.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace pass {
namespace low_precision {

ReduceBaseTransformation::ReduceBaseTransformation(const Params& params) : LayerTransformation(params) {}

bool ReduceBaseTransformation::transform(TransformationContext& context, ov::pass::pattern::Matcher& m) {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto reduce = NetworkHelper::separateInStandaloneBranch(m.get_match_root(), defaultPrecisions);
    auto dequantization = NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(reduce, defaultPrecisions));

    // prepare dequantization to propagate
    changeDequantizationValues(reduce, dequantization);

    // updatePrecision depends on type and parameters of the reduce
    const bool updatePrecision = getUpdatePrecision(reduce);
    moveDequantizationAfter(context, reduce, dequantization, updatePrecision);
    return true;
}

bool ReduceBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    const auto dequantization = NetworkHelper::getDequantization(reduce, defaultPrecisions);
    if (dequantization.empty()) {
        return false;
    }

    const auto axesConstant = ov::as_type_ptr<ov::opset1::Constant>(reduce->get_input_node_shared_ptr(1));
    if (axesConstant == nullptr) {
        return false;
    }

    // get reduced axes in normal form (without negative values)
    const auto constData = axesConstant->cast_vector<int64_t>();
    const auto inputRank = reduce->get_input_partial_shape(0).rank();
    if (inputRank.is_dynamic()) {
        return false;
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    const std::vector<size_t> axes = ov::normalize_axes(reduce->get_friendly_name(), constData, inputRank);
    OPENVINO_SUPPRESS_DEPRECATED_END

    const auto deqByReducedConst = [&](const std::shared_ptr<Node>& eltwise) {
        const auto constShape = eltwise->get_shape();

        if (!constShape.empty()) {
            for (size_t i = 0; i < constShape.size(); ++i) {
                // dequantization by reduced axis is not propagate
                if ((constShape[i] != 1ul) && std::any_of(axes.cbegin(), axes.cend(), [=](size_t elem) { return elem == i; })) {
                    return true;
                }
            }
        }
        return false;
    };

    if (dequantization.subtract != nullptr) {
        const auto normalizedSubtract = NetworkHelper::normalizeDequantizationShape(dequantization.subtract, true);
        if (normalizedSubtract == nullptr) {
            return false;
        }
        if (deqByReducedConst(normalizedSubtract)) {
            return false;
        }
    }

    const auto normalizedMultiply = NetworkHelper::normalizeDequantizationShape(dequantization.multiply);
    if (normalizedMultiply == nullptr) {
        return false;
    }
    if (deqByReducedConst(normalizedMultiply)) {
        return false;
    }

    return true;
}

void ReduceBaseTransformation::changeDequantizationValues(
    const std::shared_ptr<Node>& reduce,
    FakeQuantizeDequantization& dequantization) const {
    if (dequantization.subtract) {
        const auto newSubConstant = NetworkHelper::foldDequantizationConstant(dequantization.subtractConstant, reduce);
        replace_node(dequantization.subtractConstant, newSubConstant);
        dequantization.subtractConstant = newSubConstant;
    }

    const auto newMulConstant = NetworkHelper::foldDequantizationConstant(dequantization.multiplyConstant, reduce);
    replace_node(dequantization.multiplyConstant, newMulConstant);
    dequantization.multiplyConstant = newMulConstant;
}

bool ReduceBaseTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ov
