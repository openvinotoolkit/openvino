// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/fake_quantize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/common/ie_lpt_exception.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());

    // FakeQuantize on weights are used without dequantization ScaleShifts
    // TODO: include into the transformation pattern?
    if (NetworkHelper::onWeights(layer)) {
        return;
    }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return;
    }

    // TODO: Original LPT has a SS+FQ fusion code here; we should handle it separately

    // TODO: can we handle it by marking FQs that we wanted to exclude in RTinfo
    //       (in previous passes where quantizedFakeQuantizeNames has been populated)
    if (context.quantizedFakeQuantizeNames.find(layer->get_friendly_name()) != context.quantizedFakeQuantizeNames.end()) {
        return;
    }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) return;

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);

    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false, supportAsymmetricQuantization);
    if (dataPrecision.precision == element::undefined) {
        return;
    }

    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromQuantizationDetails(
            quantizationDetails,
            dataPrecision,
            dequantizationScales,
            dequantizationShifts);

#if 0  // TODO LPT-TO-NGRAPH
#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(dequantizationScales, dequantizationShifts);
#endif
#endif

    // Split FakeQuantize to two parts: Quantize and Dequantize

    // Quantize is represented as FakeQuantize operation, just update existing node to serve the role
    NetworkHelper::updateBlobs(layer, 3, dataPrecision.min);
    NetworkHelper::updateBlobs(layer, 4, dataPrecision.max);

    // Dequantize can be represented as Multiply+Add combination, but currently we use ScaleShiftIE.

    // Difference from legacy LPT: don't add Dequantize to not connected port; output case is covered via Result,
    // just not connected port is not used then no need to add Dequantize.

#if 0
    auto children = consumer_inputs(layer);
    std::string nameForResult = layer->get_friendly_name();
    for (auto child : children) {
        std::string nameForDequantize;
        if (child.get_node()->get_type_info().is_castable(opset1::Result::get_type_info_static())) {
            if (nameForDequantize.empty()) {
                // TODO: not a regular situation when we have more than one Result for FQ or we don't have friendly_name for FQ
            } else {
                nameForDequantize = nameForResult;
                nameForResult.clear();  // use only once
            }
        }
        auto dequantizationLayer = NetworkHelper::addScaleShiftBeforeInput(
                context,
                child,
                DequantizationDetails(dequantizationScales, dequantizationShifts, dequantizationShifts.size()),
                nameForDequantize);
        context.dequantizationLayersNames.insert(dequantizationLayer->get_friendly_name());
    }
#else
    NetworkHelper::addDequantizationAfter(
            context,
            layer->output(0),
            DequantizationDetails(dequantizationScales, dequantizationShifts, dequantizationShifts.size()));
#endif

    // Move update precision for FQ later after SS is inserted
    // It is required because we don't rely on original precision map and extract original precisions from the graph node
    // If precision for FQ is set before adding SS, then SS will derive type of its output as u8/i8 from FQ which is not correct
    if (updatePrecisions) {
        NetworkHelper::setOutDataPrecision(layer, dataPrecision.precision);
    }

    // TODO: Get rid of this.
    context.quantizedFakeQuantizeNames.insert(layer->get_friendly_name());
}

bool FakeQuantizeTransformation::isPrecisionPreserved(std::shared_ptr<Node>) const noexcept {
    return false;
}

#if 0 // TODO LPT-TO-NGRAPH
void FakeQuantizeTransformation::fuseScaleShift(TransformationContext& context, CNNLayerPtr fakeQuantizeLayer,
                                                CNNLayerPtr scaleShift) const {
    // TODO: add check if previous blobs precision is enough to store current values
    const Blob::Ptr scalesBlob = CNNNetworkHelper::getBlob(scaleShift, "weights");
    std::shared_ptr<float> scalesBufferPtr = CNNNetworkHelper::getFloatData(scalesBlob);

    const Blob::Ptr shiftsBlob = CNNNetworkHelper::getBlob(scaleShift, "biases");
    std::shared_ptr<float> shiftsBufferPtr = CNNNetworkHelper::getFloatData(shiftsBlob);

    if (scalesBlob->size() != shiftsBlob->size())
        THROW_TRANSFORMATION_EXCEPTION << "Scales and shifts values count are different for " << scaleShift->name;

    const float* shiftsBuffer = shiftsBufferPtr.get();
    const float* scalesBuffer = scalesBufferPtr.get();
    // Don't fuse when there is a negative scale, because it leads to invalid results of FQ
    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        if (scalesBuffer[i] <= 0.0f) return;
    }

    CNNLayerPtr inputLow = CNNNetworkHelper::getParent(*fakeQuantizeLayer, 1);
    CNNLayerPtr inputHigh = CNNNetworkHelper::getParent(*fakeQuantizeLayer, 2);

    Layout layout;
    size_t channelIndex;
    const DataPtr insData = scaleShift->insData[0].lock();
    if (insData == nullptr) {
        THROW_IE_LPT_EXCEPTION(*scaleShift) << "input data is absent";
    }
    const size_t inputDims = insData->getDims().size();
    switch (inputDims) {
        case 5: {
            layout = Layout::NCDHW;
            channelIndex = 1ul;
            break;
        }
        case 4: {
            layout = Layout::NCHW;
            channelIndex = 1ul;
            break;
        }
        case 3: {
            layout = Layout::BLOCKED;
            channelIndex = 1ul;
            break;
        }
        case 2: {
            layout = Layout::NC;
            channelIndex = 1ul;
            break;
        }
        case 1: {
            layout = Layout::C;
            channelIndex = 0ul;
            break;
        }
        default: THROW_TRANSFORMATION_EXCEPTION << "FakeQuantizeTransform: unexpected dimensions count " << inputDims << " in ScaleShift optimization";
    }
    std::vector<size_t> dims(inputDims, 1lu);
    dims[channelIndex] = scalesBlob->size();

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(*fakeQuantizeLayer);

    Blob::Ptr targetInputLowBufferPtr = reshapeWeightsIntervalConst(*inputLow, dims, layout);
    auto targetInputLowBuffer = CNNNetworkHelper::getFloatData(targetInputLowBufferPtr);
    Blob::Ptr targetInputHighBufferPtr = reshapeWeightsIntervalConst(*inputHigh, dims, layout);
    auto targetInputHighBuffer = CNNNetworkHelper::getFloatData(targetInputHighBufferPtr);

    for (size_t i = 0lu; i < scalesBlob->size(); ++i) {
        auto q_lo = quantizationDetails.getInputLowValue(i);
        auto q_hi = quantizationDetails.getInputHighValue(i);
        auto sc = scalesBlob->size() == 1 ? scalesBuffer[0] : scalesBuffer[i];
        auto sh = shiftsBlob->size() == 1 ? shiftsBuffer[0] : shiftsBuffer[i];
        targetInputLowBuffer.get()[i] = (q_lo - sh) / sc;
        targetInputHighBuffer.get()[i] = (q_hi - sh) / sc;
    }

    CNNNetworkHelper::fillBlobByFP32(targetInputLowBufferPtr, targetInputLowBuffer.get());
    CNNNetworkHelper::fillBlobByFP32(targetInputHighBufferPtr, targetInputHighBuffer.get());

    reshapeFakeQuantize(*fakeQuantizeLayer, dims, layout);

    CNNNetworkHelper::removeLayer(context.network, scaleShift);
    context.removeLayer(*scaleShift);
}

Blob::Ptr FakeQuantizeTransformation::reshapeWeightsIntervalConst(CNNLayer& constLayer, const std::vector<size_t>& dims,
                                                                  const Layout layout) {
    if (constLayer.blobs.size() != 1lu) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected blobs count " << constLayer.blobs.size() << " for layer " << constLayer.name;
    }
    if (constLayer.outData.size() != 1lu)
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected outputs for layer " << constLayer.name;

    auto it = constLayer.blobs.find("custom");
    if (it == constLayer.blobs.end()) THROW_TRANSFORMATION_EXCEPTION << "blob 'custom' was not found for layer " << constLayer.name;

    const Precision& srcPrecision = it->second->getTensorDesc().getPrecision();

    Blob::Ptr targetBlob = CNNNetworkHelper::makeNewBlobPtr({srcPrecision, dims, layout});
    targetBlob->allocate();
    constLayer.blobs["custom"] = targetBlob;

    constLayer.outData[0]->reshape(dims, layout);

    return targetBlob;
}

void FakeQuantizeTransformation::reshapeFakeQuantize(
        CNNLayer& fakeQuantizeLayer,
        const std::vector<size_t>& dims,
        const Layout layout) {
    DataPtr inputLowData = fakeQuantizeLayer.insData[1].lock();
    if (inputLowData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "input low interval data is absent";
    }
    inputLowData->reshape(dims, layout);

    DataPtr inputHighData = fakeQuantizeLayer.insData[2].lock();
    if (inputHighData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "input hight interval data is absent";
    }
    inputHighData->reshape(dims, layout);
}

#endif

}// namespace low_precision
}// namespace pass
}// namespace ngraph