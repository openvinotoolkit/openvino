// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/fake_quantize_decomposition.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include <low_precision/lpt_itt.hpp>
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
        if (transformation_callback(op)) {
            return false;
        }
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, "FakeQuantizeDecompositionTransformation");
    this->register_matcher(m, callback);
}

namespace fq_decomposition {

// get precision details, depends on:
// 1. FakeQuantize operation parameters (QuantizationDetails::getDetails & LayerTransformation::getPrecisionDetails)
// 2. Precisions on port
DataPrecision getDataPrecisionByOutputPortAndFakeQuantize(std::shared_ptr<opset1::FakeQuantize> layer) {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    auto precisionsAttribute = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
    if (precisionsAttribute == nullptr) {
        // TODO: explore this case in more details:
        // 1. we should not be here
        assert(true);

        // 2. not possible to get optimal precision by decomposed FakeQuantize
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(quantizationDetails);
        return DataPrecision(
            precisionDetailsAtOutputIntervals.precision,
            DataPrecision::getMinValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
            DataPrecision::getMaxValue(precisionDetailsAtOutputIntervals.precision, quantizationDetails.levels),
            precisionDetailsAtOutputIntervals.hasZeroPoint);
    }

    const auto& precisions = precisionsAttribute->get()->sharedValue->precisions;

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

        // update shared attribute to affect all operations in subgraph
        precisionsAttribute->get()->sharedValue->precisions = { precision };
    } else {
        // use only available precision
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

// get precision details, depends on:
// 1. FakeQuantize operation parameters (QuantizationDetails::getDetails & LayerTransformation::getPrecisionDetails)
// 2. Precisions on port
DataPrecision getDataPrecisionByOutputPort(std::shared_ptr<opset1::FakeQuantize> layer) {
    const size_t levels = layer->get_levels();
    const std::vector<float> outputLowValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
    const std::vector<float> outputHighValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();

    auto precisionsAttribute = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
    if (precisionsAttribute == nullptr) {
        // TODO: explore this case in more details:
        // 1. we should not be here
        assert(true);

        // 2. not possible to get optimal precision by decomposed FakeQuantize
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
            levels,
            outputLowValues,
            outputHighValues);

        return DataPrecision(
            precisionDetailsAtOutputIntervals.precision,
            DataPrecision::getMinValue(precisionDetailsAtOutputIntervals.precision, levels),
            DataPrecision::getMaxValue(precisionDetailsAtOutputIntervals.precision, levels),
            precisionDetailsAtOutputIntervals.hasZeroPoint);
    }

    const auto& precisions = precisionsAttribute->get()->sharedValue->precisions;

    ngraph::element::Type precision;
    bool hasZeroPoint;
    if (precisions.size() > 1ul) {
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
            levels,
            outputLowValues,
            outputHighValues);
        const auto foundIt = std::find(precisions.begin(), precisions.end(), precisionDetailsAtOutputIntervals.precision);

        if (foundIt == precisions.end()) {
            precision = *precisions.begin();
            hasZeroPoint = true;
        } else {
            precision = precisionDetailsAtOutputIntervals.precision;
            hasZeroPoint = precisionDetailsAtOutputIntervals.hasZeroPoint;
        }

        // update shared attribute to affect all operations in subgraph
        precisionsAttribute->get()->sharedValue->precisions = { precision };
    } else {
        // use only available precision
        precision = *precisions.begin();
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
            levels,
            outputLowValues,
            outputHighValues);
        hasZeroPoint = precisionDetailsAtOutputIntervals.precision != precision;
    }

    return DataPrecision(
        precision,
        DataPrecision::getMinValue(precision, levels),
        DataPrecision::getMaxValue(precision, levels),
        hasZeroPoint);
}

// TODO: LPT: refactor: use one way to decompose FakeQuantize
std::shared_ptr<ngraph::Node> decomposeFakeQuantize(
    MatcherPass* matcherPass,
    std::shared_ptr<opset1::FakeQuantize>& layer,
    const std::shared_ptr<IntervalsAlignmentAttribute>& intervalsAlignment,
    const DataPrecision& dataPrecision,
    const bool updatePrecisions,
    const element::Type deqPrecision) {
    std::shared_ptr<ngraph::Node> dequantize;
    if (intervalsAlignment != nullptr) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "decomposeFakeQuantize1");
        const std::vector<float> outputLowValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
        const std::vector<float> outputHighValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();

        float dequantizationMul;
        float dequantizationSub;
        float updatedOutputLowValue;
        float updatedOutputHighValue;
        const size_t levels = NetworkHelper::calculateLevels(
            dataPrecision.min,
            dataPrecision.max,
            intervalsAlignment->sharedValue->combinedInterval.low,
            intervalsAlignment->sharedValue->combinedInterval.high,
            outputLowValues[0],
            outputHighValues[0],
            dequantizationMul,
            dequantizationSub,
            updatedOutputLowValue,
            updatedOutputHighValue);

        if ((updatePrecisions == false) && (dequantizationMul == 1.f) && (dequantizationSub == 0.f)) {
            return nullptr;
        }

        //TODO: pass min levels as a parameter?
        if (levels < 2ul) {
            return nullptr;
        }

        // 2. update FakeQuantize - one time action
        std::shared_ptr<opset1::FakeQuantize> newFakeQuantizeLayer = ngraph::pass::low_precision::NetworkHelper::updateFakeQuantize(
            layer,
            updatePrecisions ? dataPrecision.precision : layer->get_output_element_type(0),
            roundf(updatedOutputLowValue),
            roundf(updatedOutputHighValue),
            false);
        matcherPass->register_new_node(newFakeQuantizeLayer);
        newFakeQuantizeLayer->set_levels(levels);

        auto dequantization = ngraph::pass::low_precision::NetworkHelper::makeDequantization(
            dequantizationMul,
            dequantizationSub,
            layer->get_output_element_type(0),
            layer->get_output_partial_shape(0),
            updatePrecisions ? dataPrecision.precision : layer->get_output_element_type(0),
            deqPrecision,
            newFakeQuantizeLayer);

        replace_node(layer, dequantization.multiply);

        std::vector<std::shared_ptr<ngraph::Node>> sourceNodes{ layer };
        std::vector<std::shared_ptr<ngraph::Node>> targetNodes{ newFakeQuantizeLayer,  dequantization.multiply };
        if (dequantization.convert != nullptr) {
            targetNodes.push_back(dequantization.convert);
        }
        if (dequantization.subtract != nullptr) {
            targetNodes.push_back(dequantization.subtract);
        }
        NetworkHelper::copyInfo(sourceNodes, targetNodes);

        dequantize = dequantization.multiply;
    } else {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "decomposeFakeQuantize2");
        // Split FakeQuantize to two parts: Quantize and Dequantize
        auto QDQ = NetworkHelper::decomposeFakeQuantize(
            as_type_ptr<opset1::FakeQuantize>(layer),
            dataPrecision.precision,
            dataPrecision.min,
            dataPrecision.max,
            dataPrecision.hasZeroPoint,
            updatePrecisions);

        const auto newFakeQuantize = std::get<0>(QDQ);
        if (newFakeQuantize == nullptr) {
            return nullptr;
        }
        matcherPass->register_new_node(newFakeQuantize);
        dequantize = std::get<1>(QDQ);
    }

    return dequantize;
}

} // namespace fq_decomposition

bool FakeQuantizeDecompositionTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto layer = as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (!NetworkHelper::isQuantizeSupported(layer)) {
        return false;
    }

    if (NetworkHelper::isFQByDynamicDimension(layer)) {
        return false;
    }

    layer = NetworkHelper::fuseConvert(layer);
    if (NetworkHelper::isConstantPath(layer)) {
        return false;
    }

    auto attribute = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
    if ((attribute == nullptr) || (attribute->get()->sharedValue->precisions.empty())) {
        return false;
    }

    const ngraph::element::Type outputPrecision = layer->get_output_element_type(0);
    if (DataPrecision::isSupported(outputPrecision)) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantizationBelow(layer);
        if (dequantization.empty()) {
            return false;
        }

        const DataPrecision expectedDataPrecision = fq_decomposition::getDataPrecisionByOutputPortAndFakeQuantize(layer);
        // TODO: need test to compose FakeQuantize
        if ((expectedDataPrecision.precision == element::undefined) || (expectedDataPrecision.precision == outputPrecision)) {
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

    DataPrecision dataPrecision = fq_decomposition::getDataPrecisionByOutputPort(layer);

    std::shared_ptr<PrecisionsAttribute> precisionsAttribute;
    {
        // TODO: LPT: return attribute (not wrapper)
        auto attributeWrapper = getAttributeFromOutput<std::shared_ptr<PrecisionsAttribute>>(layer->output(0));
        if (attributeWrapper == nullptr) {
            THROW_IE_LPT_EXCEPTION(*layer) << "PrecisionAttribute is absent";
        }
        precisionsAttribute = attributeWrapper->get();
        if (precisionsAttribute == nullptr) {
            THROW_IE_LPT_EXCEPTION(*layer) << "PrecisionAttribute is absent";
        }
    }

    std::shared_ptr<QuantizationAlignmentAttribute> quantizationAlignment;
    for (const auto& input : layer->output(0).get_target_inputs()) {
        const auto alignmentValueWrapper = low_precision::getAttribute<std::shared_ptr<QuantizationAlignmentAttribute>>(input.get_node()->shared_from_this());
        if (alignmentValueWrapper != nullptr) {
            quantizationAlignment = alignmentValueWrapper->get();
            if (quantizationAlignment->sharedValue->value) {
                break;
            }
        }
    }

    std::shared_ptr<IntervalsAlignmentAttribute> intervalsAlignment;
    {
        if ((quantizationAlignment != nullptr) && quantizationAlignment->sharedValue->value) {
            auto intervalsAlignmentWrapper = low_precision::getAttribute<std::shared_ptr<IntervalsAlignmentAttribute>>(layer);
            if (intervalsAlignmentWrapper != nullptr) {
                intervalsAlignment = intervalsAlignmentWrapper->get();
            }
        }
    }

    // FakeQuantize operations are combined in supported cascade (per tensor quantization)
    if ((intervalsAlignment != nullptr) && (intervalsAlignment->sharedValue->minLevels <= 2ul)) {
        return false;
    }

    // if IntervalsAlignment attribute is defined then, the attribute defines decomposition parameters,
    // if IntervalsAlignment attribute is not defined, then FakeQuantize operation intervals define decomposition parameters
    if (dataPrecision.precision == element::undefined) {
        element::Type precision;
        const auto levels = layer->get_levels();
        const std::vector<float> outputLowValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
        const std::vector<float> outputHighValues = as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();
        if (intervalsAlignment == nullptr) {
            // define precision by FakeQuantize intervals
            LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
                levels,
                outputLowValues,
                outputHighValues);
            const auto foundIt = std::find(
                precisionsAttribute->sharedValue->precisions.begin(),
                precisionsAttribute->sharedValue->precisions.end(),
                precisionDetailsAtOutputIntervals.precision);

            bool hasZeroPoint;
            if (foundIt == precisionsAttribute->sharedValue->precisions.end()) {
                precision = *precisionsAttribute->sharedValue->precisions.begin();
                hasZeroPoint = true;
            } else {
                precision = precisionDetailsAtOutputIntervals.precision;
                hasZeroPoint = precisionDetailsAtOutputIntervals.hasZeroPoint;
            }

            dataPrecision = DataPrecision(
                precision,
                DataPrecision::getMinValue(precision, levels),
                DataPrecision::getMaxValue(precision, levels),
                hasZeroPoint);
        } else {
            // define precision by attribute
            if (intervalsAlignment->sharedValue->preferablePrecisions.empty()) {
                // TODO: LPT: add user defined preferredPrecision
                precision = *precisionsAttribute->sharedValue->precisions.begin();
            } else {
                // TODO: LPT: add user defined preferredPrecision
                precision = *intervalsAlignment->sharedValue->preferablePrecisions.begin();
            }

            dataPrecision = DataPrecision(
                precision,
                DataPrecision::getMinValue(precision, levels),
                DataPrecision::getMaxValue(precision, levels),
                LayerTransformation::getPrecisionDetails(levels, outputLowValues, outputHighValues).precision != precision);
        }
    }

    std::shared_ptr<ngraph::Node> dequantize = fq_decomposition::decomposeFakeQuantize(
        this,
        layer,
        intervalsAlignment,
        dataPrecision,
        updatePrecisions,
        deqPrecision);
    if (dequantize == nullptr) {
        return false;
    }

    updateOutput(context, dequantize, layer);

    if (precisionsAttribute->sharedValue->precisions.size() != 1ul) {
        precisionsAttribute->sharedValue->precisions = { dataPrecision.precision };
    }

    return true;
}

bool FakeQuantizeDecompositionTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
