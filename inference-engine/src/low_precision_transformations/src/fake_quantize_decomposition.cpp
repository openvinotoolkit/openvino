// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/common/ie_lpt_exception.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"
#include "low_precision/rt_info/intervals_alignment_attribute.hpp"
#include "low_precision/rt_info/quantization_alignment_attribute.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::FakeQuantizeDecompositionTransformation, "FakeQuantizeDecompositionTransformation", 0);

FakeQuantizeDecompositionTransformation::FakeQuantizeDecompositionTransformation(const Params& params) : LayerTransformation(params) {
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (!op || transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "FakeQuantizeDecompositionTransformation");
    this->register_matcher(m, callback);
}

namespace fq_decomposition {

DataPrecision getDataPrecision(std::shared_ptr<opset1::FakeQuantize> layer) {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    auto attribute = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
    if (attribute == nullptr) {
        // TODO: explore this case in more details:
        // 1. we should not be here
        // 2. not possible to get optimal precision by decomposed FakeQuantize
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);
        return DataPrecision(
            precisionDetailsAtOutputIntervals.precision,
            DataPrecision::getMinValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
            DataPrecision::getMaxValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
            precisionDetailsAtOutputIntervals.hasZeroPoint);
    }

    const auto& precisions = attribute->get()->sharedValue->precisions;

    ngraph::element::Type precision;
    bool hasZeroPoint;
    if (precisions.size() > 1ul) {
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);
        const auto foundIt = std::find(precisions.begin(), precisions.end(), precisionDetailsAtOutputIntervals.precision);

        if (foundIt == precisions.end()) {
            precision = *precisions.begin();
            hasZeroPoint = true;
        } else {
            precision = precisionDetailsAtOutputIntervals.precision;
            hasZeroPoint = precisionDetailsAtOutputIntervals.hasZeroPoint;
        }
        attribute->get()->sharedValue->precisions = { precision };
    } else {
        precision = *precisions.begin();
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);
        hasZeroPoint = precisionDetailsAtOutputIntervals.precision != precision;
    }

    return DataPrecision(
        precision,
        DataPrecision::getMinValue(precision, quantizationDetails.levels),
        DataPrecision::getMaxValue(precision, quantizationDetails.levels),
        hasZeroPoint);
}

} // namespace fq_decomposition

bool FakeQuantizeDecompositionTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) const {
    auto layer = as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (!NetworkHelper::isQuantizeSupported(layer)) {
        return false;
    }

    layer = NetworkHelper::fuseConvert(layer);
    if (NetworkHelper::isConstantPath(layer)) {
        return false;
    }

    auto attribute = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
    if (attribute->get()->sharedValue->precisions.empty()) {
        return false;
    }

    const ngraph::element::Type precision = layer->get_output_element_type(0);
    if (DataPrecision::isSupported(precision)) {
        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantizationBelow(layer);
        if (dequantization.empty()) {
            return false;
        }

        const DataPrecision expectedDataPrecision = fq_decomposition::getDataPrecision(layer);
        // TODO: need test to compose FakeQuantize
        if ((expectedDataPrecision.precision == element::undefined) || (expectedDataPrecision.precision == precision)) {
            return false;
        }

        layer = NetworkHelper::composeFakeQuantize(layer);
        if (layer == nullptr) {
            return false;
        }
    }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return false;
    }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return false;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);

    //std::shared_ptr<QuantizationAlignmentAttribute::SharedPart::SharedValue> intervalsAlignment;
    //element::Type preferedPrecision;
    //{
    //    auto& rt = layer->get_rt_info();
    //    auto it = rt.find(ngraph::VariantWrapper<QuantizationAlignmentAttribute>::type_info.name);
    //    if (it != rt.end()) {
    //        auto attributeWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<QuantizationAlignmentAttribute>>(it->second);
    //        const QuantizationAlignmentAttribute attribute = attributeWrapper->get();
    //        intervalsAlignment = attribute.sharedPart->value->hasToBeAligned ? attribute.sharedPart->value : nullptr;
    //        preferedPrecision = attribute.sharedPart->value->preferedPrecision;
    //    }
    //}

    //DataPrecision dataPrecision;
    //{
    //    auto& rt = layer->output(0).get_rt_info();
    //    auto it = rt.find(ngraph::VariantWrapper<PrecisionsAttribute>::type_info.name);
    //    if (it != rt.end()) {
    //        auto attribute = std::dynamic_pointer_cast<ngraph::VariantWrapper<PrecisionsAttribute>>(it->second);
    //        const PrecisionsAttribute precisions = attribute->get();
    //        if (precisions.size() == 1ul) {
    //            //const bool ngraph::element::Type precision = *precisions.begin();

    //            if ((preferedPrecision == element::undefined) || (precisions.find(preferedPrecision) == precisions.end())) {
    //                // if prefered precisions are not supported then
    //                preferedPrecision = *precisions.begin();
    //            }
    //        }
    //    }
    //}

    //{
    //    PrecisionDetails precisionDetailsAtOutputIntervals = getPrecisionDetails(quantizationDetails);
    //    //const auto foundIt = std::find(precisions.begin(), precisions.end(), precisionDetailsAtOutputIntervals.precision);
    //    dataPrecision = DataPrecision(
    //        preferedPrecision,
    //        DataPrecision::getMinValue(preferedPrecision, quantizationDetails.levels),
    //        DataPrecision::getMaxValue(preferedPrecision, quantizationDetails.levels),
    //        // foundIt != precisions.end() ? precisionDetailsAtOutputIntervals.hasZeroPoint : true
    //        precisionDetailsAtOutputIntervals.precision == preferedPrecision ? precisionDetailsAtOutputIntervals.hasZeroPoint : true);
    //}

    DataPrecision dataPrecision = fq_decomposition::getDataPrecision(layer);


    std::shared_ptr<IntervalsAlignmentAttribute> intervalsAlignment;

    std::shared_ptr<ngraph::VariantWrapper<std::shared_ptr<QuantizationAlignmentAttribute>>> alignmentValue;
    for (const auto& input : layer->output(0).get_target_inputs()) {
        alignmentValue = low_precision::getAttribute<std::shared_ptr<QuantizationAlignmentAttribute>>(input.get_node()->shared_from_this());
        if ((alignmentValue != nullptr) && (alignmentValue->get()->sharedValue->value)) {
            break;
        }
    }

    if ((alignmentValue != nullptr) && alignmentValue->get()->sharedValue->value) {
        //auto& rt = layer->get_rt_info();
        //auto it = rt.find(ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>::type_info.name);
        //if (it != rt.end()) {
        //    auto attributeWrapper = std::dynamic_pointer_cast<ngraph::VariantWrapper<IntervalsAlignmentAttributePtr>>(it->second);
        //    const std::shared_ptr<IntervalsAlignmentAttribute> attribute = attributeWrapper->get();
        //    intervalsAlignment = attribute->hasToBeAligned ? attribute : nullptr;
        //}

        auto intervalsAlignmentWrapper = low_precision::getAttribute<std::shared_ptr<IntervalsAlignmentAttribute>>(layer);
        if (intervalsAlignmentWrapper != nullptr) {
            intervalsAlignment = intervalsAlignmentWrapper->get();
        }
    }

    if ((intervalsAlignment != nullptr) && (intervalsAlignment->sharedValue->minLevels >= 2ul)) {
        const auto& combinedInterval = intervalsAlignment->sharedValue->combinedInterval;
        const float maxOutputInterval = combinedInterval.high - combinedInterval.low;
        // FQ -> SUB_quantization -> MUL_quantization -[INT8]-> SUB_dequantization -> MUL_dequantization ->
        const float quantizationMul = (dataPrecision.max - dataPrecision.min) / maxOutputInterval;
        const float dequantizationMul = maxOutputInterval / (dataPrecision.max - dataPrecision.min);

        // FQ outputLowValue = dataPrecision.min * dequantizationMul - quantizationSub
        const float quantizationSub = combinedInterval.low - dataPrecision.min * dequantizationMul;
        const float dequantizationSub = std::round(-quantizationSub * quantizationMul);


        const float updatedOutputLowValue = (quantizationDetails.outputLowValues[0] - quantizationSub) * quantizationMul;
        const float updatedOutputHighValue = (quantizationDetails.outputHighValues[0] - quantizationSub) * quantizationMul;

        const size_t levels = static_cast<size_t>(fabs(roundf(updatedOutputHighValue) - roundf(updatedOutputLowValue)) + 1.0);

        //const size_t levels = NetworkHelper::calculateLevels(
        //    dataPrecision.min,
        //    dataPrecision.max,
        //    intervalsAlignment->sharedValue->intervalLow,
        //    intervalsAlignment->sharedValue->intervalHigh,
        //    quantizationDetails.outputLowValues[0],
        //    quantizationDetails.outputHighValues[0]);

        //TODO: pass min levels as a parameter?
        if (levels < 2ul) {
            return false;
        }

        // 2. update FakeQuantize - one time action
        std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
            layer,
            updatePrecisions ? dataPrecision.precision : layer->get_output_element_type(0),
            roundf(updatedOutputLowValue),
            roundf(updatedOutputHighValue),
            false);
        newFakeQuantizeLayer->set_levels(levels);

        auto dequantization = ngraph::pass::low_precision::NetworkHelper::makeDequantization(
            dequantizationMul,
            dequantizationSub,
            layer->get_output_element_type(0),
            layer->get_output_shape(0),
            updatePrecisions ? dataPrecision.precision : layer->get_output_element_type(0),
            deqPrecision,
            newFakeQuantizeLayer);

        replace_node(layer, dequantization.multiply);

        std::vector<std::shared_ptr<ngraph::Node>> sourceNodes { layer };
        std::vector<std::shared_ptr<ngraph::Node>> targetNodes { newFakeQuantizeLayer,  dequantization.multiply };
        if (dequantization.convert != nullptr) {
            targetNodes.push_back(dequantization.convert);
        }
        if (dequantization.subtract != nullptr) {
            targetNodes.push_back(dequantization.subtract);
        }
        //ngraph::copy_runtime_info(sourceNodes, targetNodes);
        NetworkHelper::copyInfo(sourceNodes, targetNodes);
    } else {
        //if (preferedPrecision == element::undefined) {
        //    if (dataPrecision.precision == element::undefined) {
        //        dataPrecision = getDataPrecision(layer, quantizationDetails, false);
        //        if (dataPrecision.precision == element::undefined) {
        //            return false;
        //        }
        //    }
        //} else {
        //    dataPrecision = DataPrecision();;
        //}

        if (dataPrecision.precision == element::undefined) {
            const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttributePtr>(layer);
            const auto precisions = precisionsAttribute == nullptr ?
                PrecisionsAttribute::defaultPrecisions :
                precisionsAttribute->get()->sharedValue->precisions;
            dataPrecision = getDataPrecision(layer, quantizationDetails, precisions);
            if (dataPrecision.precision == element::undefined) {
                return false;
            }
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
    }

    return true;
}

bool FakeQuantizeDecompositionTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
