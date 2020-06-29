// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/weightable_layer_transformation.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

namespace ngraph {
namespace pass {
namespace low_precision {

#if 0 // TODO: LPT-TO-NGRAPH

std::shared_ptr<float> broadcastActivations(const size_t batchSize, const std::vector<float>& values) {
    std::shared_ptr<float> valuesPtr(new float[values.size()], std::default_delete<float[]>());
    float* valuesRaw = valuesPtr.get();
    std::copy(values.begin(), values.end(), valuesRaw);
    return valuesPtr;
}

std::shared_ptr<float> broadcastWeights(const size_t filtersCount, const std::vector<float>& shiftsPerOuputChannel) {
    std::shared_ptr<float> valuesPtr(new float[shiftsPerOuputChannel.size()], std::default_delete<float[]>());
    float* valuesRaw = valuesPtr.get();
    std::copy(shiftsPerOuputChannel.begin(), shiftsPerOuputChannel.end(), valuesRaw);
    return valuesPtr;
}

void fillConstBlob(CNNLayer& layer, const std::vector<float>& values) {
    Blob::Ptr newBlob = CNNNetworkHelper::makeNewBlobPtr(layer.outData[0]->getTensorDesc());
    newBlob->allocate();
    CNNNetworkHelper::fillBlobByFP32(newBlob, values.data());
    layer.blobs["custom"] = newBlob;
}

#endif

WeightableLayerTransformation::WeightableLayerTransformation(const Params& params) : LayerTransformation(params) {}

bool WeightableLayerTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> layer) const {
    if (!LayerTransformation::canBeTransformed(context, layer)) {
        return false;
    }

    // Old code of this function has checks for type of inputs. Moved this code to a Convolution pattern.
    // TODO: Implement similar patterns for other weightable layers to include the checks

    const bool isDepthwiseConvolution = isDepthwise(layer);
    if (!isDepthwiseConvolution) {
        // TODO: move scale values validation to standalone method for FullyConnected & GEMM
        const auto ssNode = as_type_ptr<opset1::Multiply>(layer->input_value(0).get_node_shared_ptr());
        assert(ssNode);

        // SS takes inputs [0: data, 1: scales, 2: shifts], takes scales (index = 1)
        const auto scalesConst = as_type_ptr<opset1::Constant>(ssNode->input_value(1).get_node_shared_ptr());
        assert(scalesConst);

        // exactly cast vector as original code has a conversion;
        // TODO: optimize cast;
        // FIXME: two branches depending on real type of the constant?
        const auto scalesBuffer = scalesConst->cast_vector<float>();
        size_t scalesBufferSize = shape_size(scalesConst->get_output_shape(0));

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

#if 0
    const CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(layer, 1);
    if (parentOnWeights == nullptr) {
        return false;
    }

    OutputsDataMap outputsInfo;
    context.network.getOutputsInfo(outputsInfo);
    if (outputsInfo.find(parentOnWeights->name) != outputsInfo.end()) return false;

    const std::vector<CNNLayerPtr> weightsChildren = CNNNetworkHelper::getChildren(*parentOnWeights);
    if ((weightsChildren.size() != 1lu) || (CaselessEq<std::string>()(parentOnWeights->type, "Const") &&
                                            (parentOnWeights->outData[0]->getPrecision() != Precision::I8))) {
        return false;
    }
#endif

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

    auto tuple = decomposeFakeQuantize(fq, dataPrecision.precision, dataPrecision.min, dataPrecision.max);
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


void WeightableLayerTransformation::calculateDequantizationForSymmetric(
    std::shared_ptr<Node> convolution,
    const std::vector<float>& originalDataDequantizationScales,
    const std::vector<float>& originalDataDequantizationShifts,
    const std::vector<float>& originalWeightsDequantizationScales,
    const std::vector<float>& originalWeightsDequantizationShifts,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    std::cerr << "[ ERROR ] NOT IMPLEMENTED METHOD IS CALLED " << __FILE__ << ":" << __LINE__ << "\n";

    const size_t outputChannelCount = NetworkHelper::getOutputChannelsCount(convolution);
    dequantizationScales.resize(outputChannelCount);
    dequantizationShifts.resize(outputChannelCount);

    // TODO: Completely implement this method

#if 0 // TODO: LPT-TO-NGRAPH
    const Blob::Ptr convolutionWeightsBlob = CNNNetworkHelper::getWeights(convolution, roundQuantizedValues);
    const auto convolutionWeightsBuffer = CNNNetworkHelper::getFloatData(convolutionWeightsBlob);

    const Blob::Ptr convolutionBiasesBlob = CNNNetworkHelper::getBiases(convolution);
    const auto convolutionBiasesBuffer = convolutionBiasesBlob == nullptr ? nullptr : CNNNetworkHelper::getFloatData(convolutionBiasesBlob);


    for (size_t i = 0lu; i < dequantizationScales.size(); ++i) {
        const float originalWeightsDequantizationScale = originalWeightsDequantizationScales.size() == 0
            ? 1.0 : (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[i]);
        dequantizationScales[i] = originalDataDequantizationScales[0] * originalWeightsDequantizationScale;
    }

    const size_t inputChannelCount = CNNNetworkHelper::getInputChannelsCount(convolution);
    const size_t kernelSize = CNNNetworkHelper::getKernelSize(convolution);

    const size_t group = convolution.GetParamAsUInt("group", 1lu);
    const float originalDataDequantizationScale = originalDataDequantizationScales[0];

    const size_t outputChannelsInGroup = outputChannelCount / group;
    const size_t inputChannelsInGroup = inputChannelCount / group;
    const size_t filterSize = inputChannelsInGroup * kernelSize;

    for (size_t outputChannel = 0lu; outputChannel < outputChannelCount; ++outputChannel) {
        float sum = 0.0;
        const float originalWeightsDequantizationScale = originalWeightsDequantizationScales.size() == 0lu ?
            1.0 :
            (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[outputChannel]);
        const size_t outputChannelFilterOffset = outputChannel * filterSize;

        const size_t beginInputChannel = (outputChannel / outputChannelsInGroup) * inputChannelsInGroup;
        const size_t endInputChannel = beginInputChannel + inputChannelsInGroup;
        for (size_t inputChannel = beginInputChannel; inputChannel < endInputChannel; ++inputChannel) {
            const float originalDataDequantizationShift = originalDataDequantizationShifts[inputChannel];
            const size_t inputChannelKernelOffset = outputChannelFilterOffset + (inputChannel - beginInputChannel) * kernelSize;
            for (size_t kernelIndex = 0lu; kernelIndex < kernelSize; ++kernelIndex) {
                const float kernel = convolutionWeightsBuffer.get()[inputChannelKernelOffset + kernelIndex];
                sum += kernel * originalDataDequantizationShift * originalWeightsDequantizationScale;
            }
        }

        dequantizationShifts[outputChannel] = convolutionBiasesBuffer == nullptr
            ? sum :
            (sum + convolutionBiasesBuffer.get()[outputChannel] -
                convolutionBiasesBuffer.get()[outputChannel] * originalDataDequantizationScale * originalWeightsDequantizationScale);
    }
#endif
}


}// namespace low_precision
}// namespace pass
}// namespace ngraph
