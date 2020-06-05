// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/convolution.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_ops/multiply_add.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <cassert>

// TODO: remove after debugging
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {


#if 0 // TODO: LPT-TO-NGRAPH

void ConvolutionTransformation::calculateDequantizationForAsymmetric(
    const CNNLayer& convolution,
    const std::vector<float>& originalDataDequantizationScales,
    const std::vector<float>& originalDataDequantizationShifts,
    const std::vector<float>& dataZeroPoints,
    const std::vector<float>& originalWeightsDequantizationScales,
    const std::vector<float>& originalWeightsDequantizationShifts,
    const std::vector<float>& weightsZeroPoints,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) const {
    const size_t outputChannelCount = CNNNetworkHelper::getOutputChannelsCount(convolution);
    if (originalDataDequantizationScales.size() != outputChannelCount) {
        for (size_t i = 1ul; i < originalDataDequantizationScales.size(); ++i) {
            if (originalDataDequantizationScales[i - 1] != originalDataDequantizationScales[i])
            THROW_TRANSFORMATION_EXCEPTION << "original dequantization scales on activations have different values";
        }
    }

    dequantizationScales.resize(outputChannelCount);
    for (size_t i = 0lu; i < dequantizationScales.size(); ++i) {
        const float originalWeightsDequantizationScale = (originalWeightsDequantizationScales.size() == 0) ?
            1.0 : (originalWeightsDequantizationScales.size() == 1 ? originalWeightsDequantizationScales[0] : originalWeightsDequantizationScales[i]);
        const float originalDataDequantizationScale = (originalDataDequantizationScales.size() != dequantizationScales.size()) ?
            originalDataDequantizationScales[0] : originalDataDequantizationScales[i];
        dequantizationScales[i] = originalDataDequantizationScale * originalWeightsDequantizationScale;
    }

    dequantizationShifts.resize(outputChannelCount);

    const Blob::Ptr convolutionBiasesBlob = CNNNetworkHelper::getBiases(convolution);
    if ((convolutionBiasesBlob != nullptr) &&
        convolutionBiasesBlob->getTensorDesc().getPrecision() != Precision::FP32 &&
        convolutionBiasesBlob->getTensorDesc().getPrecision() != Precision::FP16) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected convolution biases precision "
                           << convolutionBiasesBlob->getTensorDesc().getPrecision();
    }
    const auto convolutionBiasesBuffer = convolutionBiasesBlob == nullptr ? nullptr : CNNNetworkHelper::getFloatData(convolutionBiasesBlob);

    for (size_t outputChannel = 0lu; outputChannel < outputChannelCount; ++outputChannel) {
        const float originalWeightsDequantizationScale =
            originalWeightsDequantizationScales.size() == 0lu
                ? 1.0
                : (originalWeightsDequantizationScales.size() == 1
                       ? originalWeightsDequantizationScales[0]
                       : originalWeightsDequantizationScales[outputChannel]);

        const float originalDataDequantizationScale = (outputChannel < originalDataDequantizationScales.size()) ?
            originalDataDequantizationScales[outputChannel] :
            originalDataDequantizationScales[0];

        dequantizationShifts[outputChannel] =
            convolutionBiasesBuffer == nullptr
                ? 0.0
                : convolutionBiasesBuffer.get()[outputChannel] *
                  (1.0f - originalDataDequantizationScale * originalWeightsDequantizationScale);
    }
}

#endif

ConvolutionTransformation::ConvolutionTransformation(const Params& params) : WeightableLayerTransformation(params) {}

void ConvolutionTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
            pass,
            context,
            make_op_pattern<opset1::Convolution>(
                    { make_op_label<ngraph::op::MultiplyAdd>(),
                    make_op_label<opset1::FakeQuantize>()}));
}

void ConvolutionTransformation::transform(TransformationContext &context, ngraph::pattern::Matcher &m) const {
    //return;
    auto layer = m.get_match_root();
    // Almost all checks that was in WeightableLayerTransformation now should be included into the pattern in registerMatcherIn
    if (!WeightableLayerTransformation::canBeTransformed(context, layer)) {
        return;
    }

    std::cerr << "Match convolution: " << m.get_match_root()->get_friendly_name() << "\n";

    auto scaleShiftOnData = layer->input_value(0).get_node_shared_ptr();
    auto parentOnWeights = layer->input_value(1).get_node_shared_ptr();

    /*
    std::vector<float> originalDataDequantizationScales;
    std::vector<float> originalDataDequantizationShifts;
    fillFromDequantizationLayer(scaleShiftOnData, originalDataDequantizationScales, originalDataDequantizationShifts);

    const bool isDepthwiseConvolution = isDepthwise(layer);
        // Skip checking as it is already checked in WeightableLayerTransformation::canBeTransformed
        // TODO: Acknowledge and remove this block
    }*/

    // Move Multiply from Data path immediately to the output
    // Before moving, Multiply should be decoupled and exchanged with Add in MultiplyAdd operation
    auto newMultiplyFromData = swapMultiplyAndAdd(decomposeMultiplyAdd(as_type_ptr<ngraph::op::MultiplyAdd>(scaleShiftOnData)));
    // double-check that Multiply is still scalar-like
    assert(isScalarLike(as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
    auto newMultiplyAfter = std::make_shared<opset1::Multiply>(
            layer->copy_with_new_inputs({newMultiplyFromData->input_value(0), layer->input_value(1)}),
            distillToScalar(as_type_ptr<opset1::Constant>(newMultiplyFromData->input_value(1).get_node_shared_ptr())));
    replace_node(layer, newMultiplyAfter);

    decomposeFakeQuantizeForWeightsPath(layer, supportAsymmetricQuantization);

#if 0
    std::vector<float> originalWeightsDequantizationScales;
    std::vector<float> originalWeightsDequantizationShifts;

    const DataPrecision dataPrecisionOnWeights = fillDequantizationsForWeightsPath(
            layer,
            supportAsymmetricQuantization,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts);
#endif

#if 0 // TODO: LPT-TO-NGRAPH
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(originalWeightsDequantizationScales, originalWeightsDequantizationShifts);
#endif
#endif

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;

#if 0 // TODO: LPT-TO-NGRAPH
    if (supportAsymmetricQuantization) {
        std::vector<float> dataZeroPoints(originalDataDequantizationShifts.size());
        for (size_t i = 0ul; i < originalDataDequantizationShifts.size(); ++i) {
            dataZeroPoints[i] = originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }

        std::vector<float> weightsZeroPoints(originalWeightsDequantizationShifts.size());
        for (size_t i = 0ul; i < originalWeightsDequantizationShifts.size(); ++i) {
            weightsZeroPoints[i] = originalWeightsDequantizationShifts[i] / originalWeightsDequantizationScales[i];
        }

        calculateDequantizationForAsymmetric(
            layer,
            originalDataDequantizationScales,
            originalDataDequantizationShifts,
            dataZeroPoints,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            weightsZeroPoints,
            dequantizationScales,
            dequantizationShifts);

        const Precision weightsOriginalPrecision = parentOnWeights->outData[0]->getTensorDesc().getPrecision();
        const PrecisionsInfo dataPrecisionsInfo(
            scaleShiftOnData->outData[0]->getTensorDesc().getPrecision(),
            CNNNetworkHelper::getPrecisionParent(*scaleShiftOnData));

        std::vector<float> dataShifts(originalDataDequantizationShifts.size());
        for (size_t i = 0; i < dataShifts.size(); ++i) {
            dataShifts[i] = -originalDataDequantizationShifts[i] / originalDataDequantizationScales[i];
        }

        std::vector<float> weightsShifts(originalWeightsDequantizationShifts.size());
        for (size_t i = 0; i < weightsShifts.size(); ++i) {
            weightsShifts[i] = -originalWeightsDequantizationShifts[i] / originalWeightsDequantizationScales[i];
        }

        updateToSupportAsymmetricQuantization(
            context,
            layer,
            dataPrecisionsInfo,
            dataShifts,
            PrecisionsInfo(weightsOriginalPrecision, dataPrecisionOnWeights.precision),
            weightsShifts);
    } else {
#endif
#if 0
        if (std::any_of(
            originalWeightsDequantizationShifts.begin(),
            originalWeightsDequantizationShifts.end(),
            [](const float value) { return value != 0.f; })) {
            std::cerr << "[ ERROR ]Needs support for assymentric case\n";
            return;
        }
#endif
        /*
        calculateDequantizationForSymmetric(
            layer,
            originalDataDequantizationScales,
            originalDataDequantizationShifts,
            originalWeightsDequantizationScales,
            originalWeightsDequantizationShifts,
            dequantizationScales,
            dequantizationShifts);
            */
#if 0 // TODO: LPT-TO-NGRAPH
    }
#endif

    if (this->updateBiases) {
        // Don't update biases as they are represented as a separate operation
        // TODO: Acknowledge and remove this block
        //std::vector<float> biasesShifts(dequantizationShifts.size(), 0.f);
        //updateLayerBiases(context, layer, dequantizationScales, dequantizationShifts, biasesShifts);
    }

#if 0
    NetworkHelper::removeLayer(scaleShiftOnData);
    context.removeLayer(scaleShiftOnData);

    if (weightsToConst) {
        auto weights = updatePrecisions ?
                                  NetworkHelper::quantizeWeights(parentOnWeights, roundQuantizedValues, dataPrecisionOnWeights.precision) :
                                  NetworkHelper::quantizeWeights(parentOnWeights, roundQuantizedValues);

        const std::vector<std::shared_ptr<opset1::Constant>> constLayers = NetworkHelper::transformFakeQuantizeToConst(
                context,
                parentOnWeights,
                weights,
                parentOnWeights->input_value(0).get_node()->get_friendly_name());

        if (updatePrecisions) {
            for (auto constLayer : constLayers) {
                NetworkHelper::setOutDataPrecision(constLayer, dataPrecisionOnWeights.precision);
            }
        }
    }

#endif

#if 0
    // FIXME: really required? cannot be guaranteed to match dequantizationShifts.size?
    const size_t outputChannelsCount = NetworkHelper::getOutputChannelsCount(layer);

    NetworkHelper::addDequantizationAfter(
            context,
            layer->output(0),
            DequantizationDetails(dequantizationScales, dequantizationShifts, outputChannelsCount));

#endif
}

}// namespace low_precision
}// namespace pass
}// namespace ngraph