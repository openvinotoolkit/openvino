// Copyright (C) 2020 Intel Corporation
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

bool WeightableLayerTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    if (isGroup(layer)) {
        const FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(layer);
        if (dequantization.empty()) {
            return false;
        }

        if ((dequantization.multiply != nullptr) && !FakeQuantizeDequantization::checkElementwise(dequantization.multiply)) {
            return false;
        }

        const std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(dequantization.multiply->get_input_node_shared_ptr(1));
        const Shape multiplyConstShape = multiplyConst->get_output_shape(0);
        if (!multiplyConstShape.empty() && (shape_size(multiplyConstShape) != 1ul)) {
            const size_t groupsCount = NetworkHelper::getGroupsCount(layer);
            const ngraph::Shape inputShape = layer->get_input_shape(0);
            const size_t inputChannelsInGroup = inputShape[1] / groupsCount;

            const std::vector<float> scales = multiplyConst->cast_vector<float>();
            for (size_t group = 0; group < groupsCount; ++group) {
                for (size_t i = 0; i < inputChannelsInGroup; ++i) {
                    if (scales[group * inputChannelsInGroup] != scales[group * inputChannelsInGroup + i]) {
                        return false;
                    }
                }
            }

            const ngraph::Shape outputShape = layer->get_output_shape(0);
            if ((outputShape.size() != 4ul) && (outputShape.size() != 5ul)) {
                return false;
            }
        }
    } else {
        const std::shared_ptr<opset1::Multiply> multiply = as_type_ptr<opset1::Multiply>(layer->input_value(0).get_node_shared_ptr());
        if (multiply == nullptr) {
            return false;
        }

        // SS takes inputs [0: data, 1: scales, 2: shifts], takes scales (index = 1)
        const std::shared_ptr<opset1::Constant> multiplyConst = as_type_ptr<opset1::Constant>(multiply->input_value(1).get_node_shared_ptr());
        if (multiplyConst == nullptr) {
            return false;
        }

        // exactly cast vector as original code has a conversion;
        // optimize cast:
        // two branches depending on real type of the constant?
        const auto scalesBuffer = multiplyConst->cast_vector<float>();
        size_t scalesBufferSize = shape_size(multiplyConst->get_output_shape(0));
        for (size_t i = 1lu; i < scalesBufferSize; ++i) {
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

    std::shared_ptr<opset1::Reshape> reshapeFromWeights = as_type_ptr<opset1::Reshape>(layer->input_value(1).get_node_shared_ptr());
    std::shared_ptr<opset1::FakeQuantize> fqFromWeights = as_type_ptr<opset1::FakeQuantize>(
        reshapeFromWeights == nullptr ?
        layer->input_value(1).get_node_shared_ptr() :
        layer->get_input_node_ptr(1)->get_input_node_shared_ptr(0));

    if ((fqFromWeights == nullptr) || (fqFromWeights->get_input_size() != 5ul)) {
        return false;
    }

    const Shape constOutputShape = fqFromWeights->get_input_node_ptr(3)->get_output_shape(0);
    if (fqFromWeights->get_input_node_ptr(4)->get_output_shape(0) != constOutputShape) {
        return false;
    }

    if ((constOutputShape.size() < 2ul) ||
        // Check if all dimensions of scale except the first one (which is O-Output channels dimension) are all ones
        (shape_size(constOutputShape) != constOutputShape[0]) ||
        ((constOutputShape[0] != 1ul) && (fqFromWeights->get_output_shape(0)[0] != constOutputShape[0]))) {
        return false;
    }

    return true;
}

bool WeightableLayerTransformation::isQuantized(std::shared_ptr<Node> layer, bool isReshape) const noexcept {
    auto isFakeQuantize = [](std::shared_ptr<Node> layer) {
        std::string opName = layer->get_type_name();
        return opName == "FakeQuantize";
    };

    auto parentOnWeights = layer->get_input_node_shared_ptr(1);
    std::string operationName = parentOnWeights->get_type_name();
    if (isReshape) {
        if (operationName != "Reshape") {
            return false;
        }
        parentOnWeights = parentOnWeights->get_input_node_shared_ptr(0);
        return isFakeQuantize(parentOnWeights);
    } else {
        return isFakeQuantize(parentOnWeights);
    }
}

bool WeightableLayerTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

DataPrecision WeightableLayerTransformation::decomposeFakeQuantizeForWeightsPath(std::shared_ptr<Node> node) const {
    const auto fq = getFakeQuantizeOnWeights(node);
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);
    const DataPrecision dataPrecision = getDataPrecision(fq, quantizationDetails, true);
    auto tuple = NetworkHelper::decomposeFakeQuantize(
        fq,
        dataPrecision.precision,
        dataPrecision.min,
        dataPrecision.max,
        dataPrecision.hasZeroPoint,
        updatePrecisions);

    std::shared_ptr<ngraph::Node> fqOnWeights = std::get<0>(tuple);
    if (as_type_ptr<ngraph::opset1::Constant>(fqOnWeights) == nullptr) {
        THROW_IE_LPT_EXCEPTION(*fqOnWeights) << "FakeQuantize on weights was not folded to constant";
    }

    return dataPrecision;
}

bool WeightableLayerTransformation::isGroup(const std::shared_ptr<Node>& layer) {
    if (!as_type_ptr<opset1::Convolution>(layer) && !as_type_ptr<opset1::GroupConvolution>(layer)) {
        return false;
    }

    const size_t group = NetworkHelper::getGroupsCount(layer);
    return group != 1ul;
}

bool WeightableLayerTransformation::isDepthwise(const std::shared_ptr<Node>& layer) {
    if (!as_type_ptr<opset1::Convolution>(layer) && !as_type_ptr<opset1::GroupConvolution>(layer)) {
        return false;
    }

    const size_t group = NetworkHelper::getGroupsCount(layer);
    const size_t inputChannelsCount = NetworkHelper::getInputChannelsCount(layer);
    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(layer);
    return (group == inputChannelsCount) && (inputChannelsCount == outputChannelsCount);
}

std::shared_ptr<opset1::FakeQuantize> WeightableLayerTransformation::getFakeQuantizeOnWeights(const std::shared_ptr<Node>& node) const {
    auto fq = as_type_ptr<opset1::FakeQuantize>(node->input_value(1).get_node_shared_ptr());
    // TODO: temporary workaround
    if (fq == nullptr) {
        fq = as_type_ptr<opset1::FakeQuantize>(node->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
    }

    return fq;
}

DataPrecision WeightableLayerTransformation::getDataPrecisionOnWeights(const std::shared_ptr<Node>& node) const {
    const auto fq = getFakeQuantizeOnWeights(node);
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);
    return getDataPrecision(fq, quantizationDetails, true);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
