// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/weightable_layer_transformation.hpp"
#include "low_precision/network_helper.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

namespace ngraph {
namespace pass {
namespace low_precision {

WeightableLayerTransformation::WeightableLayerTransformation(const Params& params) : LayerTransformation(params) {}

bool WeightableLayerTransformation::canConvolutionBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer,
    const std::vector<ngraph::element::Type>& defaultPrecisions) const {
    if (!WeightableLayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
    if (!canSubtractBeHandled(layer, dequantization)) {
        return false;
    }

    if (!NetworkHelper::checkZeroPoint(dequantization.subtract)) {
        return false;
    }

    if (updatePrecisions && !dequantization.empty() && !dequantization.isLowPrecision()) {
        return false;
    }

    std::shared_ptr<opset1::Reshape> reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(layer->get_input_node_shared_ptr(1));
    dequantization = reshapeFromWeights == nullptr ?
                     NetworkHelper::getDequantization(layer, defaultPrecisions, 1ul) :
                     NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions);

    if (dequantization.empty()) {
        const auto fqOnWeights = getFakeQuantizeOnWeights(layer);
        const auto dataPrecision = getDataPrecisionOnWeights(layer, defaultPrecisions);
        if ((dataPrecision.precision == ngraph::element::undefined) || (!NetworkHelper::checkZeroPoint(fqOnWeights, dataPrecision))) {
            return false;
        }
    } else {
        if (!NetworkHelper::checkZeroPoint(dequantization.subtract)) {
            return false;
        }
    }

    return true;
}

bool WeightableLayerTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    // dynamic activations rank and dynamic weights aren't supported
    if (layer->get_input_partial_shape(0).rank().is_dynamic() || layer->get_input_partial_shape(1).is_dynamic()) {
        return false;
    }

    if (isGroup(layer)) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
        if (dequantization.empty()) {
            return false;
        }

        if ((dequantization.multiply != nullptr) && !FakeQuantizeDequantization::checkElementwise(dequantization.multiply)) {
            return false;
        }

        const Shape multiplyConstShape = dequantization.multiplyConstant->get_shape();
        if (!multiplyConstShape.empty() && (shape_size(multiplyConstShape) != 1ul)) {
            const size_t groupsCount = NetworkHelper::getGroupsCount(layer);
            const PartialShape inputPShape = layer->get_input_partial_shape(0);
            const size_t inputChannelsInGroup = inputPShape[1].get_length() / groupsCount;

            const std::vector<float> scales = dequantization.multiplyConstant->cast_vector<float>();
            for (size_t group = 0; group < groupsCount; ++group) {
                for (size_t i = 0; i < inputChannelsInGroup; ++i) {
                    if (scales[group * inputChannelsInGroup] != scales[group * inputChannelsInGroup + i]) {
                        return false;
                    }
                }
            }

            const PartialShape outputPShape = layer->get_output_partial_shape(0);
            const auto rank = outputPShape.rank();
            if (rank.is_dynamic()) {
                return false;
            }

            const auto rankVal = rank.get_length();
            if ((rankVal < 3) || (rankVal > 5)) {
                return false;
            }
        }
    } else {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions);
        if (dequantization.multiply == nullptr) {
            return false;
        }

        if (dequantization.multiplyConstant == nullptr) {
            return false;
        }

        // exactly cast vector as original code has a conversion;
        // optimize cast:
        // two branches depending on real type of the constant?
        const auto scalesBuffer = dequantization.multiplyConstant->cast_vector<float>();
        size_t scalesBufferSize = shape_size(dequantization.multiplyConstant->get_shape());
        for (size_t i = 1ul; i < scalesBufferSize; ++i) {
            if (scalesBuffer[i - 1] != scalesBuffer[i]) {
                return false;
            }
        }
    }

    // Moved the rest of checks to Convolution pattern.
    // Checks are:
    //
    // [1] no other consumers for FQ sitting on weights (neither Result node, nor any others -
    // original code includes separate checks for node being output and other consumers present; for
    // ngraph it is a single check for number of consumers).
    //
    // [2] if weights is anything except a constant with data_type other than i8; this check is overriden by
    // stronger check from Convolution patter which expects FQ only on weights

    // TODO Implement similar checks in other weightable operaitons

    const std::shared_ptr<opset1::Reshape> reshapeFromWeights = ov::as_type_ptr<opset1::Reshape>(layer->get_input_node_shared_ptr(1));

    std::shared_ptr<opset1::FakeQuantize> fqFromWeights;
    if (reshapeFromWeights == nullptr) {
        fqFromWeights = ov::as_type_ptr<opset1::FakeQuantize>(layer->get_input_node_shared_ptr(1));
        if (fqFromWeights == nullptr) {
            const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer, defaultPrecisions, 1ul);
            fqFromWeights = ov::as_type_ptr<opset1::FakeQuantize>(dequantization.data.get_node_shared_ptr());
        }
    } else {
        fqFromWeights = ov::as_type_ptr<opset1::FakeQuantize>(reshapeFromWeights->get_input_node_shared_ptr(0));
        if (fqFromWeights == nullptr) {
            const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions, 0ul);
            fqFromWeights = ov::as_type_ptr<opset1::FakeQuantize>(dequantization.data.get_node_shared_ptr());
        }
    }

    if (fqFromWeights != nullptr) {
        if ((!NetworkHelper::isQuantizeSupported(fqFromWeights)) || (fqFromWeights->get_input_size() != 5ul)) {
            return false;
        }

        const auto olPShape = fqFromWeights->get_input_partial_shape(3);
        const auto ohPShape = fqFromWeights->get_input_partial_shape(4);
        if (olPShape.is_dynamic() || ohPShape.is_dynamic() || olPShape != ohPShape) {
            return false;
        }


        const auto fqOutPShape = fqFromWeights->get_output_partial_shape(0);
        const size_t outChannelsIdx = ov::is_type<opset1::ConvolutionBackpropData>(layer) ? 1ul : 0ul;
        if (fqOutPShape.rank().is_dynamic() || fqOutPShape[outChannelsIdx].is_dynamic()) {
            return false;
        }

        const Shape constShape = olPShape.to_shape();
        if (shape_size(constShape) != 1ul) {
            const size_t constChannels = constShape[outChannelsIdx];
            const size_t fqOutChannels = fqOutPShape[outChannelsIdx].get_length();
            const bool constChannelsAndFqChannelsMismatched = (constChannels != 1ul) && (fqOutChannels != constChannels);

            if ((constShape.size() <= outChannelsIdx) || (shape_size(constShape) != constChannels) || constChannelsAndFqChannelsMismatched) {
                return false;
            }
        }
    } else {
        // TODO: LPT: is it possible to share with isQuantized?
        const FakeQuantizeDequantization dequantizationOnWeights = reshapeFromWeights == nullptr ?
            NetworkHelper::getDequantization(layer, defaultPrecisions, 1ul) :
            NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions, 0ul);
        if (dequantizationOnWeights.empty()) {
            return false;
        }

        const auto weightsData = ov::as_type_ptr<opset1::Constant>(dequantizationOnWeights.data.get_node_shared_ptr());
        if (weightsData == nullptr) {
            return false;
        }

        const auto weightsDataPrecision = weightsData->get_element_type();
        if (!DataPrecision::isSupported(weightsDataPrecision)) {
            return false;
        }

        if ((dequantizationOnWeights.subtract != nullptr) && (dequantizationOnWeights.subtractConvert != nullptr)) {
            const auto subtractConstantType = dequantizationOnWeights.subtractConstant->get_element_type();
            if (subtractConstantType != weightsDataPrecision) {
                return false;
            }
        }

        const size_t outChannelsIdx = ov::is_type<opset1::ConvolutionBackpropData>(layer) ? 1ul : 0ul;
        if (dequantizationOnWeights.subtract) {
            const auto subConstShape = dequantizationOnWeights.subtractConstant->get_shape();
            if (shape_size(subConstShape) > 1ul && shape_size(subConstShape) != subConstShape[outChannelsIdx]) {
                return false;
            }
        }
        if (dequantizationOnWeights.multiply) {
            const auto mulConstShape = dequantizationOnWeights.multiplyConstant->get_shape();
            if (shape_size(mulConstShape) > 1ul && shape_size(mulConstShape) != mulConstShape[outChannelsIdx]) {
                return false;
            }
        }
    }

    return true;
}

bool WeightableLayerTransformation::isQuantizedStatic(const std::shared_ptr<const Node>& layer,
    const bool reshapeIsRequired,
    const std::vector<ngraph::element::Type>& defaultPrecisions) {
    FakeQuantizeDequantization dequantizationOnWeights;
    if (reshapeIsRequired) {
        const auto reshape = layer->get_input_node_shared_ptr(1);
        if (!ov::is_type<opset1::Reshape>(reshape)) {
            return false;
        }

        if (ov::is_type<opset1::FakeQuantize>(reshape->get_input_node_shared_ptr(0))) {
            const std::shared_ptr<opset1::FakeQuantize> fq = ov::as_type_ptr<opset1::FakeQuantize>(reshape->get_input_node_shared_ptr(0));
            return NetworkHelper::isQuantizeSupported(fq);
        }

        dequantizationOnWeights = NetworkHelper::getDequantization(reshape, defaultPrecisions, 0);
    } else if (ov::is_type<opset1::FakeQuantize>(layer->get_input_node_shared_ptr(1))) {
        const std::shared_ptr<opset1::FakeQuantize> fq = ov::as_type_ptr<opset1::FakeQuantize>(layer->get_input_node_shared_ptr(1));
        return NetworkHelper::isQuantizeSupported(fq);
    } else {
        // TODO: update NetworkHelper API later
        const std::shared_ptr<ngraph::Node> op = const_cast<ngraph::Node*>(layer.get())->shared_from_this();
        dequantizationOnWeights = NetworkHelper::getDequantization(op, defaultPrecisions, 1);
    }

    if (dequantizationOnWeights.empty()) {
        return false;
    }

    // TODO: LPT: is it possible to share with canBeTransformed?
    if (ov::is_type<opset1::Constant>(dequantizationOnWeights.data.get_node())) {
        const ngraph::element::Type weightsDataPrecision = dequantizationOnWeights.data.get_element_type();
        if (!DataPrecision::isSupported(weightsDataPrecision)) {
            return false;
        }

        if ((dequantizationOnWeights.subtract != nullptr) && (dequantizationOnWeights.subtractConvert != nullptr)) {
            const auto subtractConstantType = dequantizationOnWeights.subtractConstant->output(0).get_element_type();
            if (subtractConstantType != weightsDataPrecision) {
                return false;
            }
        }

        const size_t outChannelsShapeIndex = ov::is_type<opset1::ConvolutionBackpropData>(layer) ? 1ul : 0ul;
        if (dequantizationOnWeights.subtract) {
            const auto subConstShape = dequantizationOnWeights.subtractConstant->get_shape();
            if (shape_size(subConstShape) > 1ul && shape_size(subConstShape) != subConstShape[outChannelsShapeIndex]) {
                return false;
            }
        }
        if (dequantizationOnWeights.multiply) {
            const auto mulConstShape = dequantizationOnWeights.multiplyConstant->get_shape();
            if (shape_size(mulConstShape) > 1ul && shape_size(mulConstShape) != mulConstShape[outChannelsShapeIndex]) {
                return false;
            }
        }

        return true;
    } else if (ov::is_type<opset1::FakeQuantize>(dequantizationOnWeights.data.get_node())) {
        return true;
    }

    return false;
}

bool WeightableLayerTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

bool WeightableLayerTransformation::decomposeFakeQuantizeForWeightsPath(const std::shared_ptr<Node>& node, const size_t outChannelsShapeIndex) const {
    const auto fq = getFakeQuantizeOnWeights(node);
    if (fq == nullptr) {
        // FakeQuantize has been decomposed already
        return true;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);
    const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(fq);
    const auto precisions = precisionsAttribute.empty() ?
        defaultPrecisions :
        precisionsAttribute.as<PrecisionsAttribute>().value();

    const DataPrecision dataPrecision = getDataPrecision(fq, quantizationDetails, precisions);
    if (dataPrecision.empty()) {
        return false;
    }

    auto tuple = NetworkHelper::decomposeFakeQuantize(
        fq,
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        dataPrecision.hasZeroPoint,
        updatePrecisions,
        element::f32,
        outChannelsShapeIndex);

    std::shared_ptr<ngraph::Node> fqOnWeights = std::get<0>(tuple);
    // TODO: LPT: issue #58685
    if ((!updatePrecisions) && (fqOnWeights == nullptr)) {
        return false;
    }

    if (ov::as_type_ptr<ngraph::opset1::Constant>(fqOnWeights) == nullptr) {
        THROW_IE_LPT_EXCEPTION(*fqOnWeights) << "FakeQuantize on weights was not folded to constant";
    }

    return true;
}

bool WeightableLayerTransformation::isGroup(const std::shared_ptr<Node>& layer) {
    if (!ov::is_type<opset1::Convolution>(layer) && !ov::is_type<opset1::GroupConvolution>(layer)) {
        return false;
    }

    const size_t group = NetworkHelper::getGroupsCount(layer);
    return group != 1ul;
}

bool WeightableLayerTransformation::isDepthwise(const std::shared_ptr<Node>& layer) {
    if (!ov::as_type_ptr<opset1::Convolution>(layer) && !ov::as_type_ptr<opset1::GroupConvolution>(layer)) {
        return false;
    }

    const size_t group = NetworkHelper::getGroupsCount(layer);
    const size_t inputChannelsCount = NetworkHelper::getInputChannelsCount(layer);
    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(layer);
    return (group == inputChannelsCount) && (inputChannelsCount == outputChannelsCount);
}

std::shared_ptr<opset1::FakeQuantize> WeightableLayerTransformation::getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node) {
    auto fq = ov::as_type_ptr<opset1::FakeQuantize>(node->get_input_node_shared_ptr(1));
    // TODO: temporary workaround
    if (fq == nullptr) {
        fq = ov::as_type_ptr<opset1::FakeQuantize>(node->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
    }

    return fq;
}

DataPrecision WeightableLayerTransformation::getDataPrecisionOnWeights(
    const std::shared_ptr<Node>& node,
    const std::vector<ngraph::element::Type>& defaultPrecisions) {
    const auto fq = getFakeQuantizeOnWeights(node);
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);
    if (quantizationDetails.empty()) {
        return DataPrecision();
    }

    const auto precisionsAttribute = getAttributeFromOutput<PrecisionsAttribute>(fq);
    const auto precisions = precisionsAttribute.empty() ?
        defaultPrecisions :
        precisionsAttribute.as<PrecisionsAttribute>().value();

    return getDataPrecision(fq, quantizationDetails, precisions);
}

bool WeightableLayerTransformation::isAsymmetricOnWeights(
    const std::shared_ptr<const Node>& node,
    const std::vector<ngraph::element::Type>& defaultPrecisions) {
    const auto n = const_cast<ngraph::Node*>(node.get())->shared_from_this();

    const auto reshapeFromWeights = ngraph::as_type_ptr<ngraph::opset1::Reshape>(n->get_input_node_shared_ptr(1));
    const auto dequantization = reshapeFromWeights == nullptr ?
        NetworkHelper::getDequantization(n, defaultPrecisions, 1ul) :
        NetworkHelper::getDequantization(reshapeFromWeights, defaultPrecisions);

    if (dequantization.empty()) {
        const auto dataPrecision = WeightableLayerTransformation::getDataPrecisionOnWeights(n, defaultPrecisions);
        if (dataPrecision.empty()) {
            return false;
        }

        if (dataPrecision.hasZeroPoint) {
            return true;
        }
    } else {
        if ((dequantization.subtract != nullptr) && (NetworkHelper::optimizeSubtract(dequantization.subtract) != nullptr)) {
            return true;
        }
    }

    return false;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
