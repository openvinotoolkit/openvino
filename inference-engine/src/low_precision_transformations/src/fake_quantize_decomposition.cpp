// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeDecompositionTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

bool FakeQuantizeDecompositionTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    std::shared_ptr<opset1::FakeQuantize> layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());
    if (!NetworkHelper::isQuantizeSupported(layer)) {
        return false;
    }

    layer = NetworkHelper::fuseConvert(layer);
    if (NetworkHelper::isConstantPath(layer)) {
        // fold fq if constant just before fq and child layers aren't supported in LPT
        if (as_type<opset1::Constant>(layer->get_input_node_ptr(0))) {
            bool nextOpearionsWillBeNotHandled = true;
            for (auto output : layer->outputs()) {
                for (auto input : output.get_target_inputs()) {
                    const auto node = input.get_node();

                    if (as_type<ngraph::opset1::Reshape>(node)) {
                        for (const auto& child : NetworkHelper::consumers(node->shared_from_this())) {
                            if ((as_type_ptr<ngraph::opset1::GroupConvolution>(child)) &&
                                (paramsManager->getPrecisionsOnActivations(*child).size() != 0ul)) {
                                nextOpearionsWillBeNotHandled = false;
                                break;
                            }
                        }
                    }

                    if (paramsManager->getPrecisionsOnActivations(*input.get_node()).size() != 0ul) {
                        nextOpearionsWillBeNotHandled = false;
                        break;
                    }
                }

                if (!nextOpearionsWillBeNotHandled) {
                    break;
                }
            }

            if (nextOpearionsWillBeNotHandled) {
                const std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(layer);
                if (as_type_ptr<opset1::Constant>(resultConstant)) {
                    replace_node(layer, resultConstant);
                    return true;
                }
            }
        }
        return false;
    }

    const ngraph::element::Type precision = layer->get_output_element_type(0);
    if (DataPrecision::isSupported(precision)) {
        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantizationBelow(layer);
        if (dequantization.empty()) {
            return false;
        }

        const DataPrecision expectedDataPrecision = getDataPrecision(dequantization.multiply, quantizationDetails, false);
        if (expectedDataPrecision.precision == element::undefined) {
            return false;
        }

        if (expectedDataPrecision.precision == precision) {
            return false;
        }

        layer = NetworkHelper::composeFakeQuantize(layer);
        if (layer == nullptr) {
            return false;
        }
    }

    if (as_type<opset1::Constant>(layer->get_input_node_ptr(0))) {
        bool nextOpearionsWillBeNotHandled = true;
        for (auto output : layer->outputs()) {
            for (auto input : output.get_target_inputs()) {
                auto activations = paramsManager->getPrecisionsOnActivations(*input.get_node());
                if (paramsManager->getPrecisionsOnActivations(*input.get_node()).size() != 0ul) {
                    nextOpearionsWillBeNotHandled = false;
                    break;
                }
            }

            if (!nextOpearionsWillBeNotHandled) {
                break;
            }
        }

        if (nextOpearionsWillBeNotHandled) {
            const std::shared_ptr<ngraph::Node> resultConstant = NetworkHelper::fold_fake_quantize(layer);
            if (as_type_ptr<opset1::Constant>(resultConstant)) {
                replace_node(layer, resultConstant);
                return true;
            }
        }
    }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return false;
    }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return false;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false);
    if (dataPrecision.precision == element::undefined) {
        return false;
    }

    // Split FakeQuantize to two parts: Quantize and Dequantize
    auto QDQ = NetworkHelper::decomposeFakeQuantize(
        as_type_ptr<opset1::FakeQuantize>(layer),
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        dataPrecision.hasZeroPoint,
        updatePrecisions);

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    {
        const std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(std::get<1>(QDQ));
        const std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(multiply->get_input_node_shared_ptr(1));
        const std::vector<float> dequantizationScales = multiplyConst->cast_vector<float>();

        const std::shared_ptr<opset1::Subtract> subtract = as_type_ptr<opset1::Subtract>(multiply->get_input_node_shared_ptr(0));
        std::vector<float> dequantizationShifts;
        if (subtract != nullptr) {
            const std::shared_ptr<opset1::Constant> subtractConst = as_type_ptr<opset1::Constant>(subtract->get_input_node_shared_ptr(1));
            dequantizationShifts = subtractConst->cast_vector<float>();
        } else {
            dequantizationShifts = std::vector<float>(dequantizationScales.size());
        }

        printDequantizationValues(dequantizationScales, dequantizationShifts);
    }
#endif

    std::shared_ptr<ngraph::Node> dequantize = std::get<1>(QDQ);
    updateOutput(context, dequantize, layer);

    return true;
}

bool FakeQuantizeDecompositionTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
