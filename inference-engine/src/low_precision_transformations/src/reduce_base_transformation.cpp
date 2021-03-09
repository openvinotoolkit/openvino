// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_base_transformation.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceBaseTransformation::ReduceBaseTransformation(const Params& params) : LayerTransformation(params) {}

bool ReduceBaseTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto reduce = NetworkHelper::separateInStandaloneBranch(m.get_match_root());
    auto dequantization = NetworkHelper::normalizeDequantization(NetworkHelper::getDequantization(reduce));

    // prepare dequantization to propagate
    changeDequantizationValues(dequantization, reduce);

    // updatePrecision depends on type and parameters of the reduce
    const bool updatePrecision = getUpdatePrecision(reduce);
    moveDequantizationAfter(context, reduce, dequantization, updatePrecision);
    return true;
}

bool ReduceBaseTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    const auto dequantization = NetworkHelper::getDequantization(reduce);
    if (dequantization.multiply == nullptr) {
        return false;
    }

    const auto axesConstant = as_type_ptr<ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(1));
    if (axesConstant == nullptr) {
        return false;
    }

    // get reduced axes in normal form (without negative values)
    const auto constData = axesConstant->cast_vector<int64_t>();
    const auto inputRank = reduce->get_input_partial_shape(0).rank();
    const std::vector<size_t> axes = ngraph::normalize_axes(reduce->get_friendly_name(), constData, inputRank);

    const auto deqByReducedConst = [&](const std::shared_ptr<Node>& eltwise) {
        const auto normalizedConst = NetworkHelper::normalizeDequantizationShape(eltwise);
        const auto constShape = normalizedConst->get_shape();

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

    if (dequantization.subtract && deqByReducedConst(dequantization.subtract)) {
        return false;
    }

    if (deqByReducedConst(dequantization.multiply)) {
        return false;
    }

    return true;
}

std::shared_ptr<opset1::Constant> ReduceBaseTransformation::RemoveDimsOfDequantizationConstant(
    const std::shared_ptr<opset1::Constant>& constant,
    std::vector<size_t> reducedAxes) const {
    const auto constantShape = constant->get_shape();
    if (ngraph::shape_size(constantShape) == 1ul) {
        return NetworkHelper::toScalar(constant);
    } else {
        Shape reducedConstantShape(constantShape);
        // sort in descending order for easy removal of constant shape elements
        std::sort(reducedAxes.begin(), reducedAxes.end(), std::greater<uint64_t>());
        for (const auto& axis : reducedAxes) {
            reducedConstantShape.erase(reducedConstantShape.begin() + axis);
        }
        return opset1::Constant::create(constant->get_element_type(), reducedConstantShape, constant->cast_vector<float>());
    }
}

void ReduceBaseTransformation::changeDequantizationValues(
    FakeQuantizeDequantization& dequantization,
    const std::shared_ptr<Node>& reduce) const {
    // keepDims depends on type and parameters of the reduce
    const auto keepDims = getKeepDims(reduce);

    // get reduced axes in normal form (without negative values)
    const auto constData = as_type_ptr<ngraph::opset1::Constant>(reduce->get_input_node_shared_ptr(1))->cast_vector<int64_t>();
    const auto inputRank = reduce->get_input_partial_shape(0).rank();
    const std::vector<size_t> axes = ngraph::normalize_axes(reduce->get_friendly_name(), constData, inputRank);

    if (!keepDims) {
        if (dequantization.subtract) {
            const auto newSubConstant = RemoveDimsOfDequantizationConstant(dequantization.subtractConstant, axes);
            replace_node(dequantization.subtractConstant, newSubConstant);
            dequantization.subtractConstant = newSubConstant;
        }

        const auto newMulConstant = RemoveDimsOfDequantizationConstant(dequantization.multiplyConstant, axes);
        replace_node(dequantization.multiplyConstant, newMulConstant);
        dequantization.multiplyConstant = newMulConstant;
    }
}

bool ReduceBaseTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

bool ReduceBaseTransformation::getKeepDims(const std::shared_ptr<Node>& reduce) const {
    // default value in specification
    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
