// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/pad.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

PadTransformation::PadTransformation(const Params& params) : LayerTransformation(params) {}

void PadTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addPattern(pass,
               context,
               make_op_pattern<opset1::Pad>({
                   make_op_label<opset1::Multiply>(),
                   make_op_label<opset1::Constant>(),
                   make_op_label<opset1::Constant>(),
                   make_op_label<opset1::Constant>()}));
}

bool PadTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    if (!canBeTransformed(context, m.get_match_root())) {
        return false;
    }

    const auto pad = as_type_ptr<opset1::Pad>(NetworkHelper::separateInStandaloneBranch(m.get_match_root()));
    auto dequantization = NetworkHelper::getDequantization(pad);

    auto foldConstantIfNecessary = [&pad](const std::shared_ptr<opset1::Constant>& constant) {
        const auto constantShape = constant->get_shape();
        if (ngraph::shape_size(constantShape) == 1ul) {
            return NetworkHelper::toScalar(constant);
        }

        const auto padsBegin = pad->get_pads_begin();
        const auto padsEnd = pad->get_pads_end();

        std::vector<size_t> padsForConstantBegin(constantShape.size(), 0ul);
        std::vector<size_t> padsForConstantEnd(constantShape.size(), 0ul);
        bool foldingIsNecessary = false;

        // folding is necessary when dequantization and padding by the same dimension
        for (size_t i = 0; i < constantShape.size(); ++i) {
            if (padsBegin[i] != 0ul && constantShape[i] != 1ul) {
                foldingIsNecessary = true;
                padsForConstantBegin[i] = padsBegin[i];
            }

            if (padsEnd[i] != 0ul && constantShape[i] != 1ul) {
                foldingIsNecessary = true;
                padsForConstantEnd[i] = padsEnd[i];
            }
        }

        if (foldingIsNecessary) {
            const auto mode = pad->get_pad_mode();
            const auto beginConst = opset1::Constant::create(element::u32, { padsForConstantBegin.size() }, padsForConstantBegin);
            const auto endConst = opset1::Constant::create(element::u32, { padsForConstantEnd.size() }, padsForConstantEnd);
            const auto foldedConstant = fold<opset1::Pad>(constant, beginConst, endConst, mode);
            return as_type_ptr<opset1::Constant>(foldedConstant);
        } else {
            return constant;
        }
    };

    if (dequantization.subtract) {
        const auto normalizedSubConst = NetworkHelper::normalizeDequantizationShape(dequantization.subtract);
        const auto newSubConstant = foldConstantIfNecessary(normalizedSubConst);
        replace_node(normalizedSubConst, newSubConstant);
        dequantization.subtractConstant = newSubConstant;
    }

    const auto normalizedMulConst = NetworkHelper::normalizeDequantizationShape(dequantization.multiply);
    const auto newMulConstant = foldConstantIfNecessary(normalizedMulConst);
    replace_node(normalizedMulConst, newMulConstant);
    dequantization.multiplyConstant = newMulConstant;

    // we need to convert zero pad value in low precision
    const auto convertedZero = opset1::Constant::create(dequantization.data.get_element_type(), Shape{}, { 0 });
    pad->set_argument(3, convertedZero);

    moveDequantizationAfter(context, pad, dequantization, true);
    return true;
}

bool PadTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> op) const {
    if (!LayerTransformation::canBeTransformedSpatialDimension(context, op)) {
        return false;
    }

    const auto pad = as_type_ptr<opset1::Pad>(op);
    if (!pad) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(op);
    if (dequantization.empty()) {
        return false;
    }

    const auto mode = pad->get_pad_mode();
    // PadTransformation with "CONSTANT" requirements
    if (mode == op::PadMode::CONSTANT) {
        if (dequantization.subtract) {
            return false;
        } else {
            // non zero pad value isn't supported
            const auto constant = as_type_ptr<opset1::Constant>(pad->get_input_node_shared_ptr(3));
            const auto constantValue = constant->cast_vector<float>()[0];
            if (constantValue != 0.f) {
                return false;
            }
        }
    }

    if (mode == op::PadMode::REFLECT) {
        auto deqShape = dequantization.multiplyConstant->get_shape();
        if (ngraph::shape_size(deqShape) == 1ul) {
            return true;
        } else {
            // case when batch unspecified
            if (deqShape.size() + 1ul == pad->get_input_shape(0).size()) {
                deqShape.insert(deqShape.begin(), 1ul);
            }

            const auto padsBegin = pad->get_pads_begin();
            const auto padsEnd = pad->get_pads_end();

            // PadTransformation with "REFLECT" mode doesn't support dequantization and padding by the same dimension
            for (size_t i = 0; i < deqShape.size(); ++i) {
                if (deqShape[i] != 1ul && (padsBegin[i] != 0ul || padsEnd[i] != 0ul)) {
                    return false;
                }
            }
        }
    }

    return true;
}

bool PadTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
