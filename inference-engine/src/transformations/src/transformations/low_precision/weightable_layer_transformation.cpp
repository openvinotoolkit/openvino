// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/weightable_layer_transformation.hpp"
#include "transformations/low_precision/network_helper.hpp"

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

    const bool isDepthwiseConvolution = isDepthwise(layer);
    if (!isDepthwiseConvolution) {
        // TODO: move scale values validation to standalone method for FullyConnected & GEMM
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
        // TODO: optimize cast;
        // FIXME: two branches depending on real type of the constant?
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

bool WeightableLayerTransformation::isQuantized(std::shared_ptr<Node> layer) const noexcept {
    // TODO: not completed
    return true;
}

bool WeightableLayerTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
    return false;
}

DataPrecision WeightableLayerTransformation::decomposeFakeQuantizeForWeightsPath(
    std::shared_ptr<Node> node,
    const bool supportAsymmetricQuantization) const {
    // The first part of code analyzes FQ output parameters to select appropriate precision
    // This part doesn't use nGraph manipulations and works with raw number
    // It doesn't rely on parameters shapes and just gathers statistics, so ngraph ops are not required.

    auto fq = as_type_ptr<opset1::FakeQuantize>(node->input_value(1).get_node_shared_ptr());
    // TODO: temporary workaround
    if (fq == nullptr) {
        fq = as_type_ptr<opset1::FakeQuantize>(node->get_input_node_ptr(1)->get_input_node_shared_ptr(0));
    }


    // Obtain quantization details and decide on target precision based on dimension-less FQ parameters
    // This step is shape independent and considers FQ limits as just a set of number
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fq);
    const DataPrecision dataPrecision = getDataPrecision(fq, quantizationDetails, true, supportAsymmetricQuantization);

    // The second part of this function calculates new FQ limits and corresponding dequantization scale and shift.
    // To maintain all shapes in a consistent way, ngraph ops are used to build constant sub-expressions.

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

bool WeightableLayerTransformation::isDepthwise(std::shared_ptr<Node> layer) {
    if (!as_type_ptr<opset1::Convolution>(layer) && !as_type_ptr<opset1::GroupConvolution>(layer)) {
        return false;
    }

    const size_t group = NetworkHelper::getGroupsCount(layer);
    const size_t inputChannelsCount = NetworkHelper::getInputChannelsCount(layer);
    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(layer);
    return (group == inputChannelsCount) && (inputChannelsCount == outputChannelsCount);
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
