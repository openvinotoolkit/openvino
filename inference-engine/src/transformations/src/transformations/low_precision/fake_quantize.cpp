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

// TODO: debug only
#include <ngraph/pass/visualize_tree.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {

void FakeQuantizeTransformation::registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const {
    addSingleNodePattern<opset1::FakeQuantize>(pass, context);
}

void FakeQuantizeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    auto layer = std::dynamic_pointer_cast<opset1::FakeQuantize>(m.get_match_root());

    const std::deque<descriptor::Output> outputs = layer->get_outputs();
    const ngraph::element::Type precision = outputs.begin()->get_element_type();
    // TODO: extract to separate method (isQuantized)
    // TODO: use supported precisions
    if ((precision == ngraph::element::i8) || (precision == ngraph::element::u8)) {
        return;
    }

    // FakeQuantize on weights are used without dequantization ScaleShifts
    // TODO: include into the transformation pattern?
     if (NetworkHelper::onWeights(layer)) {
        return;
     }

    if (!QuantizationDetails::outputLayoutIsSupported(layer)) {
        return;
    }

    //// Gather Multiply from the data path
    // if (auto multiply = as_type_ptr<opset1::Multiply>(layer->input_value(0).get_node_shared_ptr())) {
    //    std::shared_ptr<opset1::Constant> constant = as_type_ptr<opset1::Constant>(multiply->input_value(0).get_node_shared_ptr());
    //    auto data = multiply->input_value(1);
    //    if (!constant) {
    //        constant = as_type_ptr<opset1::Constant>(multiply->input_value(1).get_node_shared_ptr());
    //        data = multiply->input_value(0);
    //    }

    //    if (constant) {
    //        // TODO: Check multiply consumers
    //        // TODO: verify that direct multiplication is correct
    //        auto newInputMin = fold<opset1::Divide>(layer->input_value(1), constant);
    //        auto newInputMax = fold<opset1::Divide>(layer->input_value(2), constant);
    //        // FIXME: workaround for current CPU implementation that has restrictions on shapes:
    //        auto newShape = newInputMin->get_output_shape(0);
    //        // FIXME: eshoguli: workaround for workaround to avoid 5D tensor
    //        if (newShape.size() != 4ul) {
    //            newShape.insert(newShape.begin(), 1);
    //        }
    //        newInputMin = fold_reshape<opset1::Reshape>(newInputMin, std::make_shared<opset1::Constant>(element::i64, Shape{4}, newShape), false);
    //        newInputMax = fold_reshape<opset1::Reshape>(newInputMax, std::make_shared<opset1::Constant>(element::i64, Shape{4}, newShape), false);
    //        auto newFQ = layer->copy_with_new_inputs({data, newInputMin, newInputMax, layer->input_value(3), layer->input_value(4)});
    //        replace_node(layer, newFQ);
    //        layer = as_type_ptr<opset1::FakeQuantize>(newFQ);
    //    }
    // }

    //// TODO: can we handle it by marking FQs that we wanted to exclude in RTinfo
    ////       (in previous passes where quantizedFakeQuantizeNames has been populated)
    // const std::string layerName = layer->get_friendly_name();
    // if (context.quantizedFakeQuantizeNames.find(layerName) != context.quantizedFakeQuantizeNames.end()) {
    //    return;
    // }

    if (!QuantizationDetails::isSupportedLevel(layer->get_levels())) {
        return;
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(layer);
    const DataPrecision dataPrecision = getDataPrecision(layer, quantizationDetails, false, supportAsymmetricQuantization);
    if (dataPrecision.precision == element::undefined) {
        return;
    }

#if 0 // replaced by decomposeFakeQuantize
    std::vector<float> dequantizationScales;
    std::vector<float> dequantizationShifts;
    fillFromQuantizationDetails(
            quantizationDetails,
            dataPrecision,
            dequantizationScales,
            dequantizationShifts);
#endif

#ifdef LPT_PRINT_DEQUANTIZATION_INFO
    printDequantizationValues(dequantizationScales, dequantizationShifts);
#endif

    // Split FakeQuantize to two parts: Quantize and Dequantize
    auto QDQ = decomposeFakeQuantize(
            as_type_ptr<opset1::FakeQuantize>(layer),
            dataPrecision.precision,
            dataPrecision.min,
            dataPrecision.max);

    std::vector<std::shared_ptr<ngraph::Function>> transformedModule{ context.network };
    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // To disable application of the same transform twice on the same node
    // TODO: Handle it through node property
    std::shared_ptr<opset1::FakeQuantize> quantize = as_type_ptr<opset1::FakeQuantize>(std::get<0>(QDQ));
    if (quantize != nullptr) {
        auto quantizeConvert = as_type_ptr<opset1::Convert>(quantize->get_output_target_inputs(0).begin()->get_node()->shared_from_this());

        // Remove the first Convert and built convert directly to FQ by modifying output type
        NetworkHelper::setOutDataPrecision(quantize, quantizeConvert->get_output_element_type(0));
        NetworkHelper::removeLayer(quantizeConvert);
    }

    // TODO: hardcoded
    // NetworkHelper::setOutDataPrecision(quantize, element::u8);

    auto dequantize = as_type_ptr<ngraph::Node>(std::get<1>(QDQ));
    dequantize->set_friendly_name(layer->get_friendly_name());


    // TODO: Get rid of this.
    const std::string friendlyName = layer->get_friendly_name();
    context.quantizedFakeQuantizeNames.insert(friendlyName);

    // ngraph::pass::VisualizeTree("C:\\Projects\\temp\\test.transformed").run_on_module(transformedModule);

    // std::cout << "FakeQuantizeTransformation::transform: done: " << layer->get_friendly_name() << std::endl;
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
