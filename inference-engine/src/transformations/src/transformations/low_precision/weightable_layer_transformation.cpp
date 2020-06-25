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
        std::cerr << "NOT DEPTHWISE!";
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

#if 0 // TODO: LPT-TO-NGRAPH

bool WeightableLayerTransformation::isQuantized(const CNNLayer& layer) const noexcept {
    if (!CNNNetworkHelper::isWeightsSupported(layer)) {
        return false;
    }

    const Blob::Ptr weightsBlob = CNNNetworkHelper::getWeights(layer, roundQuantizedValues);
    if ((weightsBlob == nullptr) || (!CNNNetworkHelper::isBlobPrecisionSupported(weightsBlob->getTensorDesc().getPrecision()))) {
        return false;
    }

    const Blob::Ptr biasesBlob = CNNNetworkHelper::getBiases(layer);
    if ((biasesBlob != nullptr) && (!CNNNetworkHelper::isBlobPrecisionSupported(biasesBlob->getTensorDesc().getPrecision()))) {
        return false;
    }

    const CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(layer, 1);
    if (parentOnWeights == nullptr) {
        return false;
    }

    return true;
}

bool WeightableLayerTransformation::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    return false;
}

void WeightableLayerTransformation::updateLayerBiases(
    TransformationContext& context,
    const CNNLayer& convolution,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts,
    std::vector<float>& biasesShifts) const {
    if (!std::all_of(dequantizationShifts.begin(), dequantizationShifts.end(), [](float value) { return value == 0.0; })) {
        std::shared_ptr<float> biasesBufferPtr;
        Blob::Ptr biasesBlob;
        CNNLayerPtr biasesLayer = CNNNetworkHelper::getParent(convolution, 2);
        if (biasesLayer == nullptr) {
            const std::vector<size_t> dims = CaselessEq<std::string>()(convolution.type, "Convolution") ?
                std::vector<size_t>({ dequantizationShifts.size() }) :
                std::vector<size_t>({ 1ul, dequantizationShifts.size() });
            const Layout layout = CaselessEq<std::string>()(convolution.type, "Convolution") ? Layout::C : Layout::NC;

            biasesBlob = CNNNetworkHelper::makeNewBlobPtr(TensorDesc(Precision::FP32, dims, layout));
            biasesBlob->allocate();

            biasesBufferPtr = CNNNetworkHelper::getFloatData(biasesBlob);
            float* biasesBuffer = biasesBufferPtr.get();
            std::fill(biasesBuffer, biasesBuffer + biasesBlob->size(), 0.f);

            LayerParams constLayerParams{ convolution.name + "_Biases", "Const", convolution.outData[0]->getTensorDesc().getPrecision() };
            biasesLayer = CNNNetworkHelper::addLayer(
                context,
                nullptr,
                std::make_shared<CNNLayer>(convolution),
                std::make_shared<CNNLayer>(constLayerParams));
            biasesLayer->blobs["custom"] = biasesBlob;
            biasesLayer->outData[0]->reshape(dims, layout);
        } else {
            biasesBlob = CNNNetworkHelper::getBlob(biasesLayer, "custom");
            if (biasesBlob->size() != dequantizationShifts.size()) {
                THROW_TRANSFORMATION_EXCEPTION <<
                    "dequantization shifts size " << dequantizationShifts.size() << " is not equal biases blob size " << biasesBlob->size();
            }
            biasesBufferPtr = CNNNetworkHelper::getFloatData(biasesBlob);
        }
        const float* biasesBuffer = biasesBufferPtr.get();
        std::vector<float> biases(biasesBlob->size());
        for (size_t channel = 0ul; channel < biases.size(); ++channel) {
            biases[channel] = (biasesShifts[channel] + biasesBuffer[channel]) / dequantizationScales[channel];
            dequantizationShifts[channel] = 0.0;
        }
        CNNNetworkHelper::updateBlobs(*biasesLayer, "custom", biases);
    }
}

void WeightableLayerTransformation::updateWeights(const CNNLayerPtr parent, std::vector<float>& outputLowValues,
                                                  std::vector<float>& outputHighValues) const {
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*parent);
    // TODO: refactor: move to standalone method
    switch (quantizedTensorAlignmentOnWeights) {
    case LayerTransformation::QuantizedTensorAlignment::None: {
        CNNNetworkHelper::updateBlobs(*parent, 3, outputLowValues);
        CNNNetworkHelper::updateBlobs(*parent, 4, outputHighValues);
        break;
    }
    case LayerTransformation::QuantizedTensorAlignment::UpdateIntervals:
    case LayerTransformation::QuantizedTensorAlignment::UpdateLevel: {
        THROW_TRANSFORMATION_EXCEPTION << "not implemented for weights " << quantizedTensorAlignmentOnWeights;
    }
    case LayerTransformation::QuantizedTensorAlignment::Mixed: {
        float minOutputIntervalLowValue = 0.0;
        float maxOutputIntervalHighValue = 0.0;

        for (size_t i = 0lu; i < quantizationDetails.outputLowValues.size(); ++i) {
            const float outputInterval = fabs(outputHighValues[i] - outputLowValues[i]);
            if (std::isinf(outputInterval)) {
                continue;
            }

            if (minOutputIntervalLowValue < fabs(outputLowValues[i])) {
                minOutputIntervalLowValue = fabs(outputLowValues[i]);
            }
            if (maxOutputIntervalHighValue < outputHighValues[i]) {
                maxOutputIntervalHighValue = outputHighValues[i];
            }
        }

        if (quantizationDetails.inputIntervalsCount != 1) {
            // TODO: complete later
            THROW_TRANSFORMATION_EXCEPTION << "multi input interval temporary is not supported, layer " << parent->name;
        }

        std::vector<float> inputLowValues(quantizationDetails.outputIntervalsCount);
        std::vector<float> inputHighValues(quantizationDetails.outputIntervalsCount);
        for (size_t i = 0; i < quantizationDetails.outputIntervalsCount; ++i) {
            const float minK = outputLowValues[i] == 0.0 ? 0.0 : (minOutputIntervalLowValue / fabs(outputLowValues[i]));
            inputLowValues[i] = quantizationDetails.getInputLowValue(i) * minK;
            outputLowValues[i] = roundf(outputLowValues[i] * minK);

            const float maxK =
                outputHighValues[i] == 0.0 ? 0.0 : (maxOutputIntervalHighValue / fabs(outputHighValues[i]));
            inputHighValues[i] = quantizationDetails.getInputHighValue(i) * maxK;
            outputHighValues[i] = roundf(outputHighValues[i] * maxK);
        }

        CNNNetworkHelper::updateBlobs(*parent, 1, inputLowValues);
        CNNNetworkHelper::updateBlobs(*parent, 2, inputHighValues);
        CNNNetworkHelper::updateBlobs(*parent, 3, outputLowValues);
        CNNNetworkHelper::updateBlobs(*parent, 4, outputHighValues);

        const size_t levels = static_cast<size_t>(roundf(minOutputIntervalLowValue + maxOutputIntervalHighValue + 1.0));
        parent->params["levels"] = std::to_string(levels);
        QuantizeLayer* fakeQuantizeLayer = dynamic_cast<QuantizeLayer*>(parent.get());
        if (fakeQuantizeLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "incorrect type for layer " << parent->name;
        }
        fakeQuantizeLayer->levels = levels;

        break;
    }
    default: {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected value " << quantizedTensorAlignmentOnWeights;
    }
    }
}

void WeightableLayerTransformation::updateToSupportAsymmetricQuantization(
    TransformationContext& context,
    const CNNLayer& layer,
    const PrecisionsInfo& dataPrecisionsInfo,
    std::vector<float>& dataShifts,
    const PrecisionsInfo& weightsPrecisionsInfo,
    std::vector<float>& weightsShifts) const {
    const CNNLayerPtr parentOnData = CNNNetworkHelper::getParent(layer, 0ul);
    if (parentOnData->type == "ScaleShift") {   // FIXME: it is always true
        const std::shared_ptr<float> dataConvertedInBlob = CNNNetworkHelper::convertFloatData(
            dataShifts.data(),
            dataShifts.size(),
            dataPrecisionsInfo.low);
        if (!std::all_of(dataConvertedInBlob.get(), dataConvertedInBlob.get() + dataShifts.size(), [](float value) { return value == 0.0; })) {
            createAsymmetric(context, *parentOnData, layer, dataPrecisionsInfo, dataShifts, false);
        }

        const std::shared_ptr<float> weightsConvertedInBlob = CNNNetworkHelper::convertFloatData(
            weightsShifts.data(),
            weightsShifts.size(),
            weightsPrecisionsInfo.low);
        if (!std::all_of(weightsConvertedInBlob.get(), weightsConvertedInBlob.get() + weightsShifts.size(), [](float value) { return value == 0.0; })) {
            const CNNLayerPtr parentOnWeights = CNNNetworkHelper::getParent(layer, 1ul);
            createAsymmetric(context, *parentOnWeights, layer, weightsPrecisionsInfo, weightsShifts, true);
        }
    }
}

void WeightableLayerTransformation::createAsymmetric(TransformationContext& context, const CNNLayer& parent,
                                                     const CNNLayer& child, const PrecisionsInfo& precisionsInfo,
                                                     const std::vector<float>& quantizationShifts,
                                                     const bool onWeights) const {
    if (onWeights && (parent.type != "FakeQuantize")) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected layer type on weights " << parent.type;
    }

    if (child.insData.size() < 1ul) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected layer '" << child.name << "' inputs size " << child.insData.size();
    }

    const DataPtr insData = child.insData[0].lock();
    if (insData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "insert data is absent for layer " << child.name;
    }

    if (insData->getTensorDesc().getLayout() != Layout::NC &&
        insData->getTensorDesc().getLayout() != Layout::NCHW &&
        insData->getTensorDesc().getLayout() != Layout::NCDHW) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected layout '" << insData->getTensorDesc().getLayout() << "' layer " << child.name;
    }

    LayerParams eltwiseLayerParams {child.name + "_Sub_" + parent.name, "Eltwise", precisionsInfo.original};
    std::shared_ptr<EltwiseLayer> eltwiseLayer = std::make_shared<EltwiseLayer>(eltwiseLayerParams);
    eltwiseLayer->_operation = EltwiseLayer::eOperation::Sub;
    eltwiseLayer->params["operation"] = "sub";
    CNNNetworkHelper::addLayer(context, std::make_shared<CNNLayer>(parent), std::make_shared<CNNLayer>(child),
                               eltwiseLayer);
    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision({eltwiseLayer}, precisionsInfo.original);
    }

    LayerParams constLayerParams {child.name + "_Const_" + parent.name, "Const",
                                  updatePrecisions ? precisionsInfo.low : precisionsInfo.original};
    CNNLayerPtr constLayer = std::make_shared<CNNLayer>(constLayerParams);
    constLayer = CNNNetworkHelper::addLayer(context, nullptr, eltwiseLayer, constLayer);
    if (updatePrecisions) {
        CNNNetworkHelper::setOutDataPrecision({constLayer}, precisionsInfo.low);
    }

    const TensorDesc constTensorDesc = constLayer->outData[0]->getTensorDesc();
    if (constTensorDesc.getLayout() != insData->getTensorDesc().getLayout()) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected Const layer layout " << constTensorDesc.getLayout();
    }
    const SizeVector& constDims = constTensorDesc.getDims();
    if (constDims.size() != insData->getTensorDesc().getDims().size()) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected dimension size " << constDims.size();
    }

    SizeVector dims(insData->getTensorDesc().getDims().size(), 1);
    if (onWeights) {
        dims[0] = constDims[0];
    } else {
        dims[1] = constDims[1];
    }
    constLayer->outData[0]->setDims(dims);

    fillConstBlob(*constLayer, quantizationShifts);
}

#endif

DataPrecision WeightableLayerTransformation::fillDequantizationsForWeightsPath(
        std::shared_ptr<Node> node,
        const bool supportAsymmetricQuantization,
        std::vector<float>& dequantizationScales,
        std::vector<float>& dequantizationShifts) const {
    std::cerr << "[ ERROR ] NOT IMPLEMENTED METHOD IS CALLED " << __FILE__ << ":" << __LINE__ << "\n";

    auto parent = node->input_value(1).get_node_shared_ptr();
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(as_type_ptr<opset1::FakeQuantize>(parent));
    const DataPrecision dataPrecision = getDataPrecision(parent, quantizationDetails, true, supportAsymmetricQuantization);
    fillFromQuantizationDetails(
            quantizationDetails,
            dataPrecision,
            dequantizationScales,
            dequantizationShifts);

    return dataPrecision;

    // TODO: Completely implement this method

#if 0 // TODO LPT-TO-NGRAPH
    if ((weightableLayer.type != "Convolution") && (weightableLayer.type != "FullyConnected") && (weightableLayer.type != "GEMM")) {
        THROW_TRANSFORMATION_EXCEPTION << "layer '" << weightableLayer.name << "' has unexpected type '" << weightableLayer.type << "'";
    }

    if (weightableLayer.insData.size() < 2) {
        return DataPrecision();
    }

    const DataPtr data = weightableLayer.insData[1].lock();
    if (data == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "Dequantization ScaleShift layer on weight is absent";
    }

    const CNNLayerPtr parent = CNNNetworkHelper::getParent(weightableLayer, 1);
    if (parent->type != "FakeQuantize") {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected dequantization layer type " << parent->type;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*parent);
    const DataPrecision dataPrecision = getDataPrecision(*parent, quantizationDetails, true, supportAsymmetricQuantization);
    fillFromQuantizationDetails(
        quantizationDetails,
        dataPrecision,
        dequantizationScales,
        dequantizationShifts);

    if ((!supportAsymmetricQuantization) && (
        std::any_of(dequantizationShifts.begin(), dequantizationShifts.end(), [](const float value) { return value != 0.f; }))) {
        return DataPrecision();
    }

    // TODO: going to update network: extract update weights from this method
    std::vector<float> outputLowValues(quantizationDetails.outputIntervalsCount);
    std::vector<float> outputHighValues(quantizationDetails.outputIntervalsCount);
    for (size_t i = 0; i < quantizationDetails.outputIntervalsCount; ++i) {
        if (supportAsymmetricQuantization) {
            outputLowValues[i] = dataPrecision.min;
            outputHighValues[i] = dataPrecision.max;
        } else {
            outputLowValues[i] = quantizationDetails.getOutputLowValue(i) / dequantizationScales[i];
            outputHighValues[i] = quantizationDetails.getOutputHighValue(i) / dequantizationScales[i];
        }
    }

    updateWeights(parent, outputLowValues, outputHighValues);
    return dataPrecision;
#endif
}

DataPrecision WeightableLayerTransformation::decomposeFakeQuantizeForWeightsPath(
        std::shared_ptr<Node> node,
        const bool supportAsymmetricQuantization) const {
    // The first part of code analyzes FQ output parameters to select appropriate precision
    // This part doesn't use nGraph manipulations and works with raw number
    // It doesn't rely on parameters shapes and just gathers statistics, so ngraph ops are not required.

    auto fq = as_type_ptr<opset1::FakeQuantize>(node->input_value(1).get_node_shared_ptr());
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
