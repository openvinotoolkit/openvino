// Copyright (C) 2018-2022 Intel Corporation
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
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

FakeQuantizeDecompositionTransformation::FakeQuantizeDecompositionTransformation(const Params& params) : LayerTransformation(params) {
    MATCHER_SCOPE(FakeQuantizeDecompositionTransformation);
    auto matcher = pattern::wrap_type<opset1::FakeQuantize>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(FakeQuantizeDecompositionTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

namespace fq_decomposition {
namespace {

// get precision details, depends on:
// 1. FakeQuantize operation parameters (QuantizationDetails::getDetails & LayerTransformation::getPrecisionDetails)
// 2. Precisions on port
DataPrecision getDataPrecisionByOutputPortAndFakeQuantize(std::shared_ptr<opset1::FakeQuantize> layer) {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(layer->output(0));
    if (precisionsAttribute.empty()) {
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

    const auto& precisions = precisionsAttribute.as<PrecisionsAttribute>().value();

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
        precisionsAttribute.as<PrecisionsAttribute>().value() = { precision };
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
    const std::vector<float> outputLowValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
    const std::vector<float> outputHighValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();

    auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(layer->output(0));
    if (precisionsAttribute.empty()) {
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

    const auto& precisions = precisionsAttribute.as<PrecisionsAttribute>().value();
    std::vector<element::Type> precisionsForLevels{};
    switch (levels) {
        case low_precision::levels::int16:
        case low_precision::levels::int16_narrow_range:
            precisionsForLevels = {element::u16, element::i16};
            break;
        case low_precision::levels::int32:
        case low_precision::levels::int32_narrow_range:
            precisionsForLevels = {element::u32, element::i32};
            break;
        default:
            precisionsForLevels = {element::u8, element::i8};
    }
    const auto resultPrecisions = NetworkHelper::precisionIntersection(precisions, precisionsForLevels);

    ngraph::element::Type precision;
    bool hasZeroPoint;
    if (resultPrecisions.size() > 1ul) {
        LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
            levels,
            outputLowValues,
            outputHighValues);
        const auto foundIt = std::find(resultPrecisions.begin(), resultPrecisions.end(), precisionDetailsAtOutputIntervals.precision);

        if (foundIt == resultPrecisions.end()) {
            precision = *resultPrecisions.begin();
            hasZeroPoint = true;
        } else {
            precision = precisionDetailsAtOutputIntervals.precision;
            hasZeroPoint = precisionDetailsAtOutputIntervals.hasZeroPoint;
        }

        // update shared attribute to affect all operations in subgraph
        precisionsAttribute.as<PrecisionsAttribute>().value() = { precision };
    } else {
        // use only available precision
        precision = *resultPrecisions.begin();
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
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(
    MatcherPass* matcherPass,
    std::shared_ptr<opset1::FakeQuantize>& layer,
    const ov::Any& intervalsAlignment,
    const DataPrecision& dataPrecision,
    const bool updatePrecisions,
    const element::Type deqPrecision) {
    std::shared_ptr<ngraph::Node> dequantize;
    std::shared_ptr<ngraph::Node> newFQ;

    if (!intervalsAlignment.empty()) {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "decomposeFakeQuantize1");
        const std::vector<float> outputLowValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
        const std::vector<float> outputHighValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();

        float dequantizationMul;
        float dequantizationSub;
        float updatedOutputLowValue;
        float updatedOutputHighValue;
        const size_t levels = NetworkHelper::calculateLevels(
            dataPrecision.min,
            dataPrecision.max,
            intervalsAlignment.as<IntervalsAlignmentAttribute>().value().combinedInterval.low,
            intervalsAlignment.as<IntervalsAlignmentAttribute>().value().combinedInterval.high,
            outputLowValues[0],
            outputHighValues[0],
            dequantizationMul,
            dequantizationSub,
            updatedOutputLowValue,
            updatedOutputHighValue);

        if ((updatePrecisions == false) && (dequantizationMul == 1.f) && (dequantizationSub == 0.f)) {
            return std::make_tuple(nullptr, nullptr);
        }

        //TODO: pass min levels as a parameter?
        if (levels < 2ul) {
            return std::make_tuple(nullptr, nullptr);
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

        NetworkHelper::insertDequantizationAfter(layer, dequantization.multiply, newFakeQuantizeLayer);

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
        newFQ = newFakeQuantizeLayer;
    } else {
        OV_ITT_SCOPE(FIRST_INFERENCE, itt::domains::LPT_LT, "decomposeFakeQuantize2");
        // Split FakeQuantize to two parts: Quantize and Dequantize
        auto QDQ = NetworkHelper::decomposeFakeQuantize(
            ov::as_type_ptr<opset1::FakeQuantize>(layer),
            dataPrecision.precision,
            dataPrecision.min,
            dataPrecision.max,
            dataPrecision.hasZeroPoint,
            updatePrecisions);

        const auto newFakeQuantize = std::get<0>(QDQ);
        if (newFakeQuantize == nullptr) {
            return std::make_tuple(nullptr, nullptr);
        }
        matcherPass->register_new_node(newFakeQuantize);
        dequantize = std::get<1>(QDQ);
        newFQ = newFakeQuantize;
    }

    return std::make_tuple(dequantize, newFQ);
}

} // namespace
} // namespace fq_decomposition

bool FakeQuantizeDecompositionTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher& m) {
    auto layer = ov::as_type_ptr<opset1::FakeQuantize>(m.get_match_root());
    if (!layer || !NetworkHelper::isQuantizeSupported(layer)) {
        return false;
    }

    layer = NetworkHelper::fuseConvert(layer);
    if (NetworkHelper::isConstantPath(layer)) {
        return false;
    }

    auto attribute = getAttributeFromOutput<PrecisionsAttribute>(layer->output(0));
    if (attribute.empty() || (attribute.as<PrecisionsAttribute>().value().empty())) {
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

        layer = NetworkHelper::composeFakeQuantize(layer, defaultPrecisions);
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

    PrecisionsAttribute precisionsAttribute(defaultPrecisions);
    {
        // TODO: LPT: return attribute (not wrapper)
        auto attributeWrapper = getAttributeFromOutput<PrecisionsAttribute>(layer->output(0));
        if (attributeWrapper.empty()) {
            THROW_IE_LPT_EXCEPTION(*layer) << "PrecisionAttribute is absent";
        }
        precisionsAttribute = attributeWrapper.as<PrecisionsAttribute>();
    }

    ov::Any quantizationAlignment;
    for (const auto& input : layer->output(0).get_target_inputs()) {
        quantizationAlignment = low_precision::getAttribute<QuantizationAlignmentAttribute>(input.get_node()->shared_from_this());
        if (!quantizationAlignment.empty()) {
            if (quantizationAlignment.as<QuantizationAlignmentAttribute>().value()) {
                break;
            }
        }
    }

    ov::Any intervalsAlignment;
    if (!quantizationAlignment.empty() && quantizationAlignment.as<QuantizationAlignmentAttribute>().value()) {
        intervalsAlignment = low_precision::getAttribute<IntervalsAlignmentAttribute>(layer);
    }

    // FakeQuantize operations are combined in supported cascade (per tensor quantization)
    if (!intervalsAlignment.empty() && (intervalsAlignment.as<IntervalsAlignmentAttribute>().value().minLevels <= 2ul)) {
        return false;
    }

    // if IntervalsAlignment attribute is defined then, the attribute defines decomposition parameters,
    // if IntervalsAlignment attribute is not defined, then FakeQuantize operation intervals define decomposition parameters
    if (dataPrecision.precision == element::undefined) {
        element::Type precision;
        const auto levels = layer->get_levels();
        const std::vector<float> outputLowValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(3))->cast_vector<float>();
        const std::vector<float> outputHighValues = ov::as_type_ptr<opset1::Constant>(layer->get_input_node_shared_ptr(4))->cast_vector<float>();
        if (intervalsAlignment.empty()) {
            // define precision by FakeQuantize intervals
            LayerTransformation::PrecisionDetails precisionDetailsAtOutputIntervals = LayerTransformation::getPrecisionDetails(
                levels,
                outputLowValues,
                outputHighValues);
            const auto foundIt = std::find(
                precisionsAttribute.value().begin(),
                precisionsAttribute.value().end(),
                precisionDetailsAtOutputIntervals.precision);

            bool hasZeroPoint;
            if (foundIt == precisionsAttribute.value().end()) {
                precision = *precisionsAttribute.value().begin();
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
            if (intervalsAlignment.as<IntervalsAlignmentAttribute>().value().preferablePrecisions.empty()) {
                // TODO: LPT: add user defined preferredPrecision
                precision = *precisionsAttribute.value().begin();
            } else {
                // TODO: LPT: add user defined preferredPrecision
                precision = *intervalsAlignment.as<IntervalsAlignmentAttribute>().value().preferablePrecisions.begin();
            }

            dataPrecision = DataPrecision(
                precision,
                DataPrecision::getMinValue(precision, levels),
                DataPrecision::getMaxValue(precision, levels),
                LayerTransformation::getPrecisionDetails(levels, outputLowValues, outputHighValues).precision != precision);
        }
    }

    auto QDQ = fq_decomposition::decomposeFakeQuantize(
        this,
        layer,
        intervalsAlignment,
        dataPrecision,
        updatePrecisions,
        deqPrecision);

    std::shared_ptr<ngraph::Node> dequantize = std::get<0>(QDQ);
    std::shared_ptr<ngraph::Node> newFakeQuantize = std::get<1>(QDQ);
    if (dequantize == nullptr || newFakeQuantize == nullptr) {
        return false;
    }

    updateOutput(context, dequantize, newFakeQuantize);

    if (precisionsAttribute.value().size() != 1ul) {
        precisionsAttribute.value() = { dataPrecision.precision };
    }

    return true;
}

bool FakeQuantizeDecompositionTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}
} // namespace low_precision
} // namespace pass
} // namespace ngraph
