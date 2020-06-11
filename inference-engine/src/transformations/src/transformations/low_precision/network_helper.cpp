// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <transformations/low_precision/network_helper.hpp>
#include <ngraph_ops/multiply_add.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cassert>

#include <ngraph/rt_info.hpp>

// #include <ie_common.h>
// #include <precision_utils.h>
// #include "cnn_network_impl.hpp"
// #include "ie_util_internal.hpp"
// #include "ie_parallel.hpp"
#include <transformations/low_precision/common/ie_lpt_exception.hpp>

namespace ngraph {
namespace pass {
namespace low_precision {


// Return true if `type` can be castable to at least one of `type`
bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types) {
    for (auto another : types) {
        if (type.is_castable(another)) {
            return true;
        }
    }
    return false;
}


// Collect and return a vector with all nodes that consumes any of the `node` output
std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node) {
    std::vector<Input<Node>> result;
    for (const auto& output_port : node->outputs()) {
        for (const auto &input : output_port.get_target_inputs()) {
            result.push_back(input);
        }
    }
    return result;
}

std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node) {
    auto inputs = consumer_inputs(node);
    std::vector<std::shared_ptr<Node>> result(inputs.size());
    std::transform(inputs.begin(), inputs.end(), result.begin(), [](Input<Node> input){ return input.get_node()->shared_from_this(); });
    return result;
}

#if 0 // TODO LPT-TO-NGRAPH
CNNLayerPtr NetworkHelper::getLayer(const ICNNNetwork& network, const std::string& layerName) {
    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : layers) {
        if (layer->name == layerName) {
            return layer;
        }
    }

    return nullptr;
}

Blob::Ptr NetworkHelper::makeNewBlobPtr(const TensorDesc& desc) {
    Blob::Ptr newBlob;
    if (desc.getPrecision() == Precision::FP32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(desc);
    else if (desc.getPrecision() == Precision::FP16)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP16>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::U8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I32>::value_type>(desc);
    else
        THROW_TRANSFORMATION_EXCEPTION << "Unsupported transformation precision: " << desc.getPrecision();

    return newBlob;
}

void NetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, float value) {
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
        THROW_TRANSFORMATION_EXCEPTION << "blob '" << blobName << "' was not found in layer " << layer.name;
    }
    const auto& existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    Blob::Ptr newBlob = makeNewBlobPtr(existingBlobTensorDesc);

    newBlob->allocate();
    fillBlobByFP32(newBlob, value);
    layer.blobs[existingBlobIt->first] = newBlob;
}


void NetworkHelper::invertFakeQuantize(const CNNLayer& fakeQuantize) {
    if (fakeQuantize.type != "FakeQuantize") {
        THROW_TRANSFORMATION_EXCEPTION << "invalid layer type " << fakeQuantize.type;
    }
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
    const size_t valuesCount =
        std::max(quantizationDetails.inputLowValues.size(), quantizationDetails.outputLowValues.size());
    std::vector<float> inputLowValues(valuesCount);
    std::vector<float> inputHightValues(valuesCount);
    std::vector<float> outputLowValues(valuesCount);
    std::vector<float> outputHighValues(valuesCount);
    bool wasInverted = false;
    for (size_t i = 0ul; i < valuesCount; ++i) {
        if ((quantizationDetails.getInputLowValue(i) > quantizationDetails.getInputHighValue(i)) &&
            (quantizationDetails.getOutputLowValue(i) > quantizationDetails.getOutputHighValue(i))) {
            inputLowValues[i] = quantizationDetails.getInputHighValue(i);
            inputHightValues[i] = quantizationDetails.getInputLowValue(i);
            outputLowValues[i] = quantizationDetails.getOutputHighValue(i);
            outputHighValues[i] = quantizationDetails.getOutputLowValue(i);
            wasInverted = true;
        } else {
            inputLowValues[i] = quantizationDetails.getInputLowValue(i);
            inputHightValues[i] = quantizationDetails.getInputHighValue(i);
            outputLowValues[i] = quantizationDetails.getOutputLowValue(i);
            outputHighValues[i] = quantizationDetails.getOutputHighValue(i);
        }
    }

    if (wasInverted) {
        NetworkHelper::updateBlobs(fakeQuantize, 1, inputLowValues);
        NetworkHelper::updateBlobs(fakeQuantize, 2, inputHightValues);
        NetworkHelper::updateBlobs(fakeQuantize, 3, outputLowValues);
        NetworkHelper::updateBlobs(fakeQuantize, 4, outputHighValues);
    }
}

#endif
#if 0 // TODO LPT-TO-NGRAPH

void NetworkHelper::updateBlobs(std::shared_ptr<opset1::FakeQuantize> layer, int constLayerIndex,
                                   const std::vector<float>& values) {
    auto constant = std::dynamic_pointer_cast<opset1::Constant>(layer->get_input_node_shared_ptr(constLayerIndex));
    if (!constant) {
        THROW_TRANSFORMATION_EXCEPTION << "Expected constant at " << constLayerIndex << " input for FakeQuantize node" << *layer;
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
            if (existingBlobTensorDesc.getDims().size() != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
            if (existingBlobTensorDesc.getDims().size() != 4) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : blobLayer->outData) {
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    blobLayer->blobs[existingBlobIt->first] = newBlob;

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

#endif
#if 0 // TODO LPT-TO-NGRAPH

void NetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values) {
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
        THROW_TRANSFORMATION_EXCEPTION << "custom blob was not found ";
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
            if (existingBlobTensorDesc.getDims().size() != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
            if (existingBlobTensorDesc.getDims().size() != 4) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : layer.outData) {
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    layer.blobs[existingBlobIt->first] = newBlob;

    if ((blobName == "weights") || (blobName == "biases")) {
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(&layer);
        if (weightableLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "layer '" << layer.name << "' with blob name '" << blobName << "' is not weightable";
        }
        if (blobName == "weights") {
            weightableLayer->_weights = newBlob;
        } else if (blobName == "biases") {
            weightableLayer->_biases = newBlob;
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected blob name '" << blobName << "' for layer " << layer.name;
        }
    }

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

#endif

void NetworkHelper::updateBlobs(std::shared_ptr<opset1::FakeQuantize> quantizeLayer, int constLayerIndex, float value) {
    auto constant = std::dynamic_pointer_cast<opset1::Constant>(quantizeLayer->get_input_node_shared_ptr(constLayerIndex));
    if (!constant) {
        THROW_TRANSFORMATION_EXCEPTION << "Expected constant at " << constLayerIndex << " input for FakeQuantize node" << *quantizeLayer;
    }

    auto new_constant = std::make_shared<opset1::Constant>(constant->get_output_element_type(0), constant->get_output_shape(0), value);
    copy_runtime_info(constant, new_constant);
    replace_node(constant, new_constant);
}

int NetworkHelper::onWeightsInDepth(std::shared_ptr<Node> layer) {
    const std::vector<std::shared_ptr<Node>> children = consumers(layer);
    for (std::shared_ptr<Node> child : children) {
        if ((layer->get_type_info().is_castable(opset1::Convolution::get_type_info_static()) ||
             layer->get_type_info().is_castable(opset1::GroupConvolution::get_type_info_static()) ||
             layer->get_type_info().is_castable(opset1::MatMul::get_type_info_static())) &&
             child->inputs().size() >= 2lu) {
            const std::vector<std::shared_ptr<Node>> parents = getParentsRecursivelyExceptTypes(child, {}, 1);
            for (auto parent : parents) {
                // ???
                if (parent->get_friendly_name() == layer->get_friendly_name()) {
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(child);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

bool NetworkHelper::onWeights(std::shared_ptr<Node> layer) {
    const int result = onWeightsInDepth(layer);
    return result == 1;
}

#if 0 // TODO LPT-TO-NGRAPH

size_t NetworkHelper::getIndex(const CNNLayer& layer) {
    const std::vector<CNNLayerPtr> children = NetworkHelper::getChildren(layer);
    if (children.size() != 1) {
        THROW_TRANSFORMATION_EXCEPTION << "not supported";
    }

    for (size_t i = 0; i < children[0]->insData.size(); ++i) {
        const DataPtr insData = children[0]->insData[i].lock();
        if (insData == nullptr) {
            continue;
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if ((parent != nullptr) && (parent->name == layer.name)) {
            return i;
        }
    }

    THROW_TRANSFORMATION_EXCEPTION << "not found";
}

#endif

std::vector<std::shared_ptr<ngraph::opset1::Constant>> NetworkHelper::transformFakeQuantizeToConst(TransformationContext& context,
                                                                        std::shared_ptr<Node> fakeQuantize,
                                                                        std::shared_ptr<opset1::Constant> weights,
                                                                        const std::string& constLayerName) {
    // TODO: update context by deleting removed layer and adding new layer if really needed
    // TODO: set proper name for a constant
    std::vector<std::shared_ptr<ngraph::opset1::Constant>> result{weights};
    copy_runtime_info(fakeQuantize, weights);
    replace_node(fakeQuantize, weights);
    return result;
#if 0 // TODO: LPT-TO-NGRAPH
    std::vector<CNNLayerPtr> constLayersToRemove;
    constLayersToRemove.reserve(fakeQuantize->insData.size());

    for (const DataWeakPtr& insDataWeak : fakeQuantize->insData) {
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input data for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input layer for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        if (!CaselessEq<std::string>()(parent->type, "Const") || (parent->insData.size() != 0lu)) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected FakeQuantize input layer type " << parent->type << " for layer '"
                               << fakeQuantize->name << "' is nullable";
        }

        constLayersToRemove.push_back(parent);
    }

    for (const CNNLayerPtr& parent : constLayersToRemove) {
        NetworkHelper::removeLayer(context.network, parent);
        context.removeLayer(*parent);
    }

    if (fakeQuantize->outData.size() != 1lu) {
        THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize " << fakeQuantize->name << " has several outputs";
    }

    const DataPtr outData = fakeQuantize->outData[0];
    if (outData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "FakeQuantize output data is nullable";
    }

    // const Precision precision = outData->getPrecision();
    const auto inputTo = outData->getInputTo();
    std::vector<CNNLayerPtr> constLayers;
    for (auto it : inputTo) {
        const CNNLayerPtr child = it.second;
        if (child == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "child layer for FakeQuantize " << fakeQuantize->name << " is nullable";
        }

        constLayers.push_back(
            NetworkHelper::addConstBetween(context.network, fakeQuantize, child, weights, constLayerName));
    }

    NetworkHelper::removeLayer(context.network, fakeQuantize);
    context.removeLayer(*fakeQuantize);

    return constLayers;

#endif
}

/*
void NetworkHelper::setOutDataPrecision(std::shared_ptr<Node>, const element::Type& precision) {
    for (const DataPtr& data : layer.outData) {
        data->setPrecision(precision);
    }
    // std::cout << "Just set precision " << precision << " for node " << layer.name << "\n";
}
*/

#if 0 // TODO LPT-TO-NGRAPH

void NetworkHelper::setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision) {
    for (const CNNLayerPtr layer : layers) {
        setOutDataPrecision(*layer, precision);
    }
}

void NetworkHelper::setOutDataPrecision(const CNNLayer& beginLayer, const size_t branchWithEndBeforeLayer,
                                           const CNNLayer& endBeforeLayer, const Precision& precision) {
    CNNLayerPtr child = std::make_shared<CNNLayer>(beginLayer);
    while (child->name != endBeforeLayer.name) {
        NetworkHelper::setOutDataPrecision(*child, precision);
        std::vector<CNNLayerPtr> children = NetworkHelper::getChildren(*child);
        if (child->name == beginLayer.name) {
            if (branchWithEndBeforeLayer >= children.size()) {
                THROW_TRANSFORMATION_EXCEPTION << "branch with end before layer is out of children count " << children.size();
            }
            child = children[branchWithEndBeforeLayer];
        } else {
            if (children.size() != 1) {
                THROW_TRANSFORMATION_EXCEPTION << "not supported";
            }

            child = children[0];
        }
    }
}

#endif

bool NetworkHelper::IsChild(
        const std::vector<std::shared_ptr<Node>>& children,
        const std::vector<NodeTypeInfo>& layerTypes,
        const std::vector<NodeTypeInfo>& ignoreLayerTypes) {
    for (auto child : children) {
        for (auto layer_type : layerTypes) {
            if (child->get_type_info().is_castable(layer_type)) {
                return true;
            }
        }
        for (auto ignore_type : ignoreLayerTypes) {
            if (child->get_type_info().is_castable(ignore_type)) {
                if (child->outputs().size() != 1) {
                    return true;
                }
                if (IsChild(consumers(child), layerTypes, ignoreLayerTypes)) {
                    return true;
                }
            }
        }
    }
    return false;
}

size_t NetworkHelper::getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights) {
    if (layer->outputs().size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " doesn't have output tensors";
    }

    if (layer->outputs().size() > 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer->get_friendly_name() << " has too many output tensors, expected one";
    }

    PartialShape shape = layer->get_output_partial_shape(0);
    if (shape.rank() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in output of " << layer->get_friendly_name() << " layer on weights";
    }
    if (isOnWeights) {
        return shape[0].get_length();
    } else {
        if (shape.rank() == 1) {
            return shape[0].get_length();
        }
        return shape[1].get_length();
    }
}

#if 0 // TODO LPT-TO-NGRAPH

std::vector<CNNLayerPtr> NetworkHelper::getLayers(const CNNLayer& parent, const CNNLayer& child) {
    std::vector<CNNLayerPtr> layers;
    CNNLayerPtr tmpChild = std::make_shared<CNNLayer>(child);
    while (tmpChild != nullptr) {
        const std::vector<CNNLayerPtr> parents = NetworkHelper::getParents(*tmpChild);
        for (const CNNLayerPtr tmpParent : parents) {
            if (tmpParent->name == parent.name) {
                return layers;
            }
        }

        if (parents.size() == 0) {
            THROW_TRANSFORMATION_EXCEPTION << "not found";
        }

        if (parents.size() != 1ul) {
            THROW_TRANSFORMATION_EXCEPTION << "not supported";
        }

        layers.push_back(parents[0]);
        tmpChild = parents[0];
    }
    return layers;
}

Blob::Ptr NetworkHelper::getBlob(CNNLayer* layer, const std::string& blobName) {
    if (layer == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "layer is nullable";
    }
    if (layer->blobs.empty()) {
        THROW_TRANSFORMATION_EXCEPTION << "Layer '" << layer->name << "' does not have any blob";
    }
    if (blobName.empty() && (layer->blobs.size() != 1)) {
        THROW_TRANSFORMATION_EXCEPTION << "several blobs";
    }
    Blob::Ptr blob = blobName.empty() ? layer->blobs.begin()->second : layer->blobs[blobName];
    return blob;
}

Blob::Ptr NetworkHelper::getBlob(CNNLayerPtr layer, const std::string& blobName) {
    return getBlob(layer.get(), blobName);
}

std::shared_ptr<float> NetworkHelper::getFloatData(const Blob::Ptr& srcBlob) {
    if (srcBlob == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid blob";
    }

    const auto& precision = srcBlob->getTensorDesc().getPrecision();
    if (!isBlobPrecisionSupported(precision)) {
        THROW_TRANSFORMATION_EXCEPTION << "precision '" << precision << "' is not supported";
    }

    const size_t dataSize = srcBlob->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
        const float* srcData = srcBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::FP16) {
        const short* srcData = srcBlob->buffer().as<short*>();
        PrecisionUtils::f16tof32Arrays(floatPtr.get(), srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::U8) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I32) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I64) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::U64) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::U64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return floatPtr;
}

bool NetworkHelper::isBlobPrecisionSupported(const Precision precision) {
    return (precision == Precision::FP32) ||
        (precision == Precision::FP16) ||
        (precision == Precision::I8) ||
        (precision == Precision::U8) ||
        (precision == Precision::I32) ||
        (precision == Precision::I64) ||
        (precision == Precision::U64);
}

std::shared_ptr<float> NetworkHelper::getFloatData(const CNNLayerPtr& layer, const std::string& blobName) {
    const Blob::Ptr blob = getBlob(layer, blobName);
    if (blob == nullptr) THROW_TRANSFORMATION_EXCEPTION << "Could not find blob '" << blobName << "' for layer " << layer->name;

    return getFloatData(blob);
}

void NetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData) {
    if (dstBlob == nullptr) THROW_TRANSFORMATION_EXCEPTION << "Invalid blob";

    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, dstData);
    } else if (precision == Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        PrecisionUtils::f32tof16Arrays(dstData, srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::U8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::I32) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i]));
        }
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

std::shared_ptr<float> NetworkHelper::convertFloatData(const float* srcData, const size_t dataSize,
                                                          const Precision precision) {
    std::shared_ptr<float> dstData(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
        std::copy(srcData, srcData + dataSize, dstData.get());
    } else if (precision == Precision::FP16) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] = PrecisionUtils::f16tof32(PrecisionUtils::f16tof32(srcData[i]));
        }
    } else if (precision == Precision::I8) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::U8) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::I32) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i])));
        }
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return dstData;
}

void NetworkHelper::fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData) {
    Blob::Ptr blob = getBlob(layer, blobName);
    return fillBlobByFP32(blob, srcData);
}

void NetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, float value) {
    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::fill(dstData, dstData + dataSize, value);
    } else if (precision == Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        const short s_value = PrecisionUtils::f32tof16(value);
        std::fill(dstData, dstData + dataSize, s_value);
    } else if (precision == Precision::I8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I8>::value_type>(value));
    } else if (precision == Precision::U8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::U8>::value_type>(value));
    } else if (precision == Precision::I32) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I32>::value_type>(value));
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

CNNLayerPtr NetworkHelper::getParent(const CNNLayer& layer, const size_t index, const std::string& ignoreLayerType) {
    if (index >= layer.insData.size()) {
        return nullptr;
    }

    DataPtr inputLayerData = layer.insData[index].lock();
    if (inputLayerData == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "input data is absent";
    }

    CNNLayerPtr inputLayer;
    do {
        inputLayer = inputLayerData->getCreatorLayer().lock();
        if (!inputLayer) {
            THROW_TRANSFORMATION_EXCEPTION << "input is absent";
        }

        if (inputLayer->type != ignoreLayerType) {
            break;
        }

        if (inputLayer->insData.size() == 0) {
            inputLayer = nullptr;
            break;
        }

        if (inputLayer->insData.size() != 1) {
            THROW_TRANSFORMATION_EXCEPTION << "too much branches";
        }

        inputLayerData = inputLayer->insData[0].lock();
        if (inputLayerData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input data is absent";
        }
    } while (true);

    return inputLayer;
}

std::vector<CNNLayerPtr> NetworkHelper::getParents(const CNNLayer& layer, const std::string& exceptionLayerName) {
    std::vector<CNNLayerPtr> parents;
    for (const DataWeakPtr insDataWeak : layer.insData) {
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input data is absent";
        }

        CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "input layer is absent";
        }

        if (exceptionLayerName.empty() || parent->name != exceptionLayerName) {
            parents.push_back(parent);
        }
    }
    return parents;
}

#endif

std::vector<std::shared_ptr<Node>> NetworkHelper::getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes,
        const int portIndex) {
    std::vector<std::shared_ptr<Node>> parents;
    size_t i = 0ul;
    for (auto input : layer->inputs()) {
        if ((portIndex == -1) || (portIndex == i)) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            if (is_castable_to_one_of(parent->get_type_info(), exceptionLayerTypes)) {
                const std::vector<std::shared_ptr<Node>> tmpParents = getParentsRecursivelyExceptTypes(parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

size_t NetworkHelper::getInputChannelsCount(std::shared_ptr<Node> layer) {
    if (layer->get_input_size() == 0) {
        THROW_TRANSFORMATION_EXCEPTION << "There are no input layers";
    }

    PartialShape shape = layer->get_input_partial_shape(0);
    if (shape.rank().get_length() <= 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid dimensions count (0) in input of " << layer->get_friendly_name();
    }

    return shape[1].get_length();
}

size_t NetworkHelper::getGroupsCount(std::shared_ptr<Node> layer) {
    if (as_type_ptr<opset1::Convolution>(layer)) {
        return 1;
    } else if (auto group_convolution = as_type_ptr<opset1::GroupConvolution>(layer)) {
        return layer->get_input_shape(0)[0];    // input weights for opset1::GC is in format GOI..., see the specification
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid layer type of " << layer->get_friendly_name() << "; expected Convolutino or GroupConvolution";
    }
}

#if 0 // TODO LPT-TO-NGRAPH

size_t NetworkHelper::getParamOutput(const CNNLayer& layer) {
    if (!layer.CheckParamPresence("output")) {
        THROW_TRANSFORMATION_EXCEPTION << "convolution parameter 'output' is absent";
    }
    return layer.GetParamAsUInt("output");
}

size_t NetworkHelper::getKernelSize(const CNNLayer& layer) {
    if (!layer.CheckParamPresence("kernel")) {
        THROW_TRANSFORMATION_EXCEPTION << "convolution parameter 'kernel' is absent";
    }
    const auto dims = layer.GetParamAsUInts("kernel");
    if (dims.size() == 2) {
        return dims[0] * dims[1];
    } else if (dims.size() == 3) {
        return dims[0] * dims[1] * dims[2];
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "kernel dimensions are not correct";
    }
}

void NetworkHelper::renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName) {
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected network type";
    }

    netImpl->renameLayer(currentName, newName);
}

CNNLayerPtr NetworkHelper::addLayer(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const CNNLayerPtr newLayer) {
    DataPtr outData;
    Precision precision;
    if (parent != nullptr) {
        // Searching the connection between the layers
        int l1_out_i = 0;
        if (child != nullptr) {
            for (; l1_out_i < parent->outData.size(); l1_out_i++) {
                if (parent->outData[l1_out_i]->getInputTo().find(child->name) !=
                    parent->outData[l1_out_i]->getInputTo().end()) {
                    break;
                }
            }
        }
        if (l1_out_i == parent->outData.size()) {
            if (child != nullptr)
                THROW_TRANSFORMATION_EXCEPTION << "Can't find layer " << child->name << " among layer " << parent->name << " outputs";
            else
                THROW_TRANSFORMATION_EXCEPTION << "Layer '" << parent->name << "' has invalid output";
        }

        outData = parent->outData[l1_out_i];
        precision = context.getOriginalLayerPrecision(parent->name, outData->getName());
        IE_SUPPRESS_DEPRECATED_START
        if (precision == Precision::UNSPECIFIED) {
            if (child != nullptr)
                precision = child->precision;
            else if (context.network.getPrecision() != Precision::MIXED)
                precision = context.network.getPrecision();
            else
                precision = Precision::FP32;
        }
        IE_SUPPRESS_DEPRECATED_END
    } else {
        // TODO: FIXME
        precision = Precision::FP32;
        outData = nullptr;
    }
    addLayerToCNNNetworkAfterData(outData, newLayer, child != nullptr ? child->name : "", context.network);

    NetworkHelper::setOutDataPrecision(*newLayer, precision);
    return newLayer;
}

void NetworkHelper::replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target) {
    CNNNetworkImpl* networkImpl = dynamic_cast<CNNNetworkImpl*>(&context.network);
    networkImpl->removeLayer(source->name);

    std::vector<CNNLayerPtr> parents = NetworkHelper::getParents(*source);
    for (CNNLayerPtr parent : parents) {
        for (size_t outDataIndex = 0ul; outDataIndex < parent->outData.size(); ++outDataIndex) {
            const DataPtr outData = parent->outData[outDataIndex];
            std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
            inputTo[source->name] = target;
            target->insData.push_back(outData);
        }
    }

    const std::vector<CNNLayerPtr> children = NetworkHelper::getChildren(*source);

    target->outData.resize(source->outData.size());
    for (size_t outDataIndex = 0ul; outDataIndex < source->outData.size(); ++outDataIndex) {
        const DataPtr outData = source->outData[outDataIndex];
        networkImpl->removeData(outData->getName());

        DataPtr newOutData(new Data(outData->getName(), outData->getTensorDesc()));
        newOutData->getCreatorLayer() = target;
        target->outData[outDataIndex] = newOutData;
        networkImpl->addData(newOutData->getName().c_str(), newOutData);

        std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (const auto it : inputTo) {
            const CNNLayerPtr child = it.second;
            newOutData->getInputTo().emplace(it.first, child);

            for (const CNNLayerPtr& child : children) {
                for (size_t insDataIndex = 0ul; insDataIndex < child->insData.size(); ++insDataIndex) {
                    const DataPtr insData = child->insData[insDataIndex].lock();
                    if (insData == nullptr) {
                        THROW_IE_LPT_EXCEPTION(*child) << "insert data " << insDataIndex << " is absent";
                    }

                    const CNNLayerPtr parent = insData->getCreatorLayer().lock();
                    if (parent == nullptr) {
                        THROW_IE_LPT_EXCEPTION(*child) << "parent layer for insert data " << insDataIndex << " is absent";
                    }
                    if (parent->name == source->name) {
                        const auto it = target->outData[outDataIndex];
                        child->insData[insDataIndex] = newOutData;
                    }
                }
            }
        }
        outData->getInputTo().clear();
    }

    IE_SUPPRESS_DEPRECATED_START
    context.network.addLayer(target);
    IE_SUPPRESS_DEPRECATED_END
}

#endif

// Assumin tensor in NC... layout, append necessary number of 1s to shape to align it to a give rank
Shape alignShapeForChannelDim(const Shape& shape, Rank rank) {
    assert(shape.size() == 1);
    assert(rank.is_static());
    Shape result = shape;
    result.resize(rank.get_length() - 1, 1);
    return result;
}

std::shared_ptr<Node> NetworkHelper::addScaleShiftBeforeInput(TransformationContext& context,
                                                   const Input<Node>& input,
                                                   const DequantizationDetails& dequantizationDetails,
                                                   const std::string& name) {
#if 0 // TODO: LPT-TO-NGRAPH: on-the-fly fuse into the next SS should be implemented as a separate pass; don't do it now
    if (child && (child->type == "ScaleShift") && (NetworkHelper::getParents(*child).size() == 1)) {
        auto scalesIt = child->blobs.find("weights");
        if (scalesIt == child->blobs.end()) {
            THROW_TRANSFORMATION_EXCEPTION << "weights for layer " << child->name << " was not found";
        }
        const std::shared_ptr<float> scales = NetworkHelper::getFloatData(scalesIt->second);
        std::vector<float> updatedScales(scalesIt->second->size());
        for (size_t i = 0ul; i < updatedScales.size(); ++i) {
            updatedScales[i] = scales.get()[i] * dequantizationDetails.scales[i];
        }
        NetworkHelper::updateBlobs(*child, "weights", updatedScales);

        auto shiftsIt = child->blobs.find("biases");
        if (shiftsIt != child->blobs.end()) {
            const std::shared_ptr<float> shifts = NetworkHelper::getFloatData(shiftsIt->second);
            std::vector<float> updatedShifts(shiftsIt->second->size());
            for (size_t i = 0ul; i < updatedShifts.size(); ++i) {
                updatedShifts[i] = scales.get()[i] * dequantizationDetails.shifts[i] + shifts.get()[i];
            }
            NetworkHelper::updateBlobs(*child, "biases", updatedShifts);
        }

        return child;
    }
#endif

    auto parent = input.get_source_output().get_node_shared_ptr();
    std::string layerName = name.empty() ? (parent->get_friendly_name() + "_ScaleShift_" + input.get_node()->get_friendly_name()) : name;

    element::Type ssPrecision = context.getOriginalLayerPrecision(parent->get_friendly_name(), input.get_source_output().get_index());
    // TODO: LPT-TO-NGRAPH, not sure that it covers all valid cases
    if (ssPrecision == element::undefined) {
        ssPrecision = input.get_element_type();
    }

    auto scaleConst = std::make_shared<opset1::Constant>(
            ssPrecision,
            alignShapeForChannelDim(Shape{dequantizationDetails.channelsCount}, input.get_partial_shape().rank()),
            dequantizationDetails.scales);
    auto shiftConst = std::make_shared<opset1::Constant>(
            ssPrecision,
            alignShapeForChannelDim(Shape{dequantizationDetails.channelsCount}, input.get_partial_shape().rank()),
            dequantizationDetails.shifts);

    auto ssLayer = std::make_shared<ngraph::op::MultiplyAdd>(input.get_source_output(), scaleConst, shiftConst);

    input.get_source_output().remove_target_input(input); // Disconnect source output from input of interest
    input.replace_source_output(ssLayer->output(0)); // Connect input of interest to just created new node ssLayer output

    NetworkHelper::setOutDataPrecision(ssLayer, ssPrecision);
    return ssLayer;
}

void NetworkHelper::addDequantizationAfter(TransformationContext& context,
                                                              const Output<Node>& output,
                                                              const DequantizationDetails& dequantizationDetails) {
    auto node = output.get_node_shared_ptr();

    // TODO: provide consumer_inputs for a single output port and replace here
    auto children = consumer_inputs(node);

    std::string nameForResult = node->get_friendly_name();
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
        auto dequantizationLayer = addScaleShiftBeforeInput(
                context,
                child,
                dequantizationDetails,
                nameForDequantize);
        context.dequantizationLayersNames.insert(dequantizationLayer->get_friendly_name());
    }
}

#if 0 // TODO LPT-TO-NGRAPH

CNNLayerPtr NetworkHelper::addConstBetween(ICNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
                                              const Blob::Ptr customBlob, const std::string& name) {
    if (layer1 == nullptr)
        THROW_TRANSFORMATION_EXCEPTION << "First layer is nullable";
    // Searching the connection between the layers
    int l1_out_i = 0;
    if (layer2 != nullptr) {
        for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
            if (layer1->outData[l1_out_i]->getInputTo().find(layer2->name) !=
                layer1->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
    }

    if (l1_out_i == layer1->outData.size()) {
        if (layer2 != nullptr)
            THROW_TRANSFORMATION_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
        else
            THROW_TRANSFORMATION_EXCEPTION << "Layer " << layer1->name << " has invalid outputs";
    }

    DataPtr outData = layer1->outData[l1_out_i];

    std::string layerName = name.empty() ? layer1->name + "_Const" : name;
    CNNLayerPtr layer(new CNNLayer({layerName, "Const", customBlob->getTensorDesc().getPrecision()}));

    addLayerToCNNNetworkAfterData(outData, layer, layer2 != nullptr ? layer2->name : "", net);
    layer->blobs.emplace("custom", customBlob);
    layer->outData[0]->setPrecision(customBlob->getTensorDesc().getPrecision());
    return layer;
}

void NetworkHelper::addLayerToCNNNetworkAfterData(
    DataPtr parentOutData,
    CNNLayer::Ptr layer,
    const std::string& nextLayerName,
    ICNNNetwork& net) {
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "unexpected network type";
    }

    CNNLayerPtr nextLayer;
    if (!nextLayerName.empty()) {
        netImpl->getLayerByName(nextLayerName.c_str(), nextLayer, nullptr);
    }

    if (layer && (nextLayerName.empty() || (parentOutData == nullptr) ||
                  (parentOutData->getInputTo().find(nextLayerName) != parentOutData->getInputTo().end()))) {
        auto getTensorDesc = [](CNNLayerPtr& nextLayer) {
            const DataPtr insData = nextLayer->insData[0].lock();
            if (insData == nullptr) {
                THROW_IE_LPT_EXCEPTION(*nextLayer) << "insert data is absent";
            }
            return insData->getTensorDesc();
        };

        const TensorDesc& parentTensorDesc = parentOutData != nullptr ? parentOutData->getTensorDesc() : getTensorDesc(nextLayer);
        DataPtr newEdgeAfterLayer(new Data(layer->name, parentTensorDesc));
        newEdgeAfterLayer->setName(layer->name);
        newEdgeAfterLayer->getCreatorLayer() = layer;
        newEdgeAfterLayer->getInputTo().clear();

        CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
        if (netImpl == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "unexpected network type";
        }
        netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
        IE_SUPPRESS_DEPRECATED_START
        netImpl->addLayer(layer);
        IE_SUPPRESS_DEPRECATED_END

        if (parentOutData != nullptr) {
            parentOutData->getInputTo()[layer->name] = layer;
            layer->insData.push_back(parentOutData);
        }
        layer->outData.push_back(newEdgeAfterLayer);

        if (!nextLayerName.empty()) {
            // CNNLayerPtr nextLayer = parentOutData->getInputTo()[nextLayerName];
            newEdgeAfterLayer->getInputTo()[nextLayerName] = nextLayer;
            if (parentOutData != nullptr) {
                parentOutData->getInputTo().erase(nextLayerName);
                for (size_t i = 0; i < nextLayer->insData.size(); i++) {
                    if (nextLayer->insData[i].lock() == parentOutData) {
                        nextLayer->insData[i] = newEdgeAfterLayer;
                    }
                }
            } else {
                // TODO: why new?
                nextLayer->insData.push_back(newEdgeAfterLayer);
            }
        } else {
            CNNLayerPtr parent = parentOutData->getCreatorLayer().lock();
            if (parent == nullptr) {
                THROW_TRANSFORMATION_EXCEPTION << "parent data is absent";
            }
            netImpl->removeOutput(parent->name);
            netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
            netImpl->addOutput(layer->name);
        }
    } else {
        THROW_TRANSFORMATION_EXCEPTION << "Invalid argument";
    }
}

void NetworkHelper::fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales,
                                        const float* shifts) {
    if (layer == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "ScaleShiftLayer is nullable";
    }

    layer->_weights = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_weights->allocate();
    fillBlobByFP32(layer->_weights, scales);
    layer->blobs["weights"] = layer->_weights;

    layer->_biases = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_biases->allocate();
    fillBlobByFP32(layer->_biases, shifts);
    layer->blobs["biases"] = layer->_biases;
}

std::vector<CNNLayerPtr> NetworkHelper::getChildren(const CNNLayer& layer, const std::string& exceptionLayerName) {
    // TODO: replaced by consumers, but not for all cases - should be implemented anyway as it has exceptionLayerName (consumers doesn't)
    std::vector<CNNLayerPtr> children;
    for (const DataPtr outData : layer.outData) {
        const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
            CNNLayerPtr child = it->second;
            if (exceptionLayerName.empty() || child->name != exceptionLayerName) {
                children.push_back(child);
            }
        }
    }
    return children;
}

#endif

std::vector<std::shared_ptr<Node>> NetworkHelper::getChildrenRecursivelyExceptTypes(
        std::shared_ptr<Node> layer, const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes) {
    std::vector<std::shared_ptr<Node>> children;
    for (auto child : consumers(layer)) {
        if (is_castable_to_one_of(child->get_type_info(), exceptionLayerTypes)) {
            const std::vector<std::shared_ptr<Node>> tmpChildren = getChildrenRecursivelyExceptTypes(child, exceptionLayerTypes);
            children.insert(children.end(), tmpChildren.begin(), tmpChildren.end());
        }
        children.push_back(child);
    }
    return children;
}

#if 0 // TODO LPT-TO-NGRAPH

void NetworkHelper::checkConstWithBlobs(const CNNLayerPtr layer) {
    if (layer->type != "Const") {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void NetworkHelper::checkQuantizeOnWeights(const CNNLayerPtr layer) {
    if (layer->type != "FakeQuantize") {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 0) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 5) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void NetworkHelper::updateInput(CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData) {
    if (!CaselessEq<std::string>()(layer->type, "Input")) {
        return;
    }

    InputInfo::Ptr inputInfo = network->getInput(layer->name);
    if (inputInfo->name() == layer->name) {
        inputInfo->setInputData(outData);
    }
}

size_t NetworkHelper::disconnectLayers(CNNNetworkImpl* network, const CNNLayerPtr& parentLayer,
                                          const CNNLayerPtr& childLayer) {
    bool wasFound = false;
    for (auto dataIt = parentLayer->outData.begin(); dataIt != parentLayer->outData.end(); ++dataIt) {
        auto data = *dataIt;
        for (auto inputIt = data->getInputTo().begin(); inputIt != data->getInputTo().end(); ++inputIt) {
            auto currentChildLayer = inputIt->second;
            if (currentChildLayer == nullptr) {
                THROW_TRANSFORMATION_EXCEPTION << "Output layer for '" << parentLayer->name << "'is absent";
            }
            if (currentChildLayer->name == childLayer->name) {
                const DataPtr dataToRemove = network->getData(data->getName().c_str());
                if (!dataToRemove) {
                    THROW_TRANSFORMATION_EXCEPTION << "there is not data to remove";
                }

                data->getInputTo().erase(inputIt);
                wasFound = true;
                break;
            }
        }

        if (wasFound) {
            break;
        }
    }
    if (!wasFound) {
        THROW_TRANSFORMATION_EXCEPTION << "Output layer '" << childLayer->name << "' was not found for '" << parentLayer->name
                           << "'";
    }

    wasFound = false;
    for (auto it = childLayer->insData.begin(); it != childLayer->insData.end(); ++it) {
        auto data = it->lock();
        if (data == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "Input layer data for '" << childLayer->name << "'is absent";
        }
        auto currentParentLayer = data->getCreatorLayer().lock();
        if (currentParentLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "Input layer for '" << childLayer->name << "'is absent";
        }
        if (currentParentLayer->name == parentLayer->name) {
            childLayer->insData.erase(it);
            wasFound = true;
            break;
        }
    }
    if (!wasFound) {
        THROW_TRANSFORMATION_EXCEPTION << "Input layer '" << parentLayer->name << "' was not found for '" << childLayer->name
                           << "'";
    }
    return 0;
}

size_t NetworkHelper::getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer) {
    for (size_t index = 0; index < childLayer->insData.size(); ++index) {
        DataPtr currentParenData = childLayer->insData[index].lock();
        if (currentParenData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "parent layer data is absent";
        }
        CNNLayerPtr currentParrentLayer = currentParenData->getCreatorLayer().lock();
        if (currentParrentLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "parent layer is absent";
        }
        if (currentParrentLayer->name == parentLayer->name) {
            return index;
        }
    }

    THROW_TRANSFORMATION_EXCEPTION << "parent layer was not found";
}

#endif

void NetworkHelper::removeLayer(std::shared_ptr<Node> layer) {
    ngraph::replace_output_update_name(layer->output(0), layer->input_value(0));
}

#if 0 // TODO: LPT-TO-NGRAPH

bool NetworkHelper::isWeightsSupported(const CNNLayer& layer) noexcept {
    if (layer.insData.size() > 1) {
        CNNLayerPtr weightsLayer = NetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr)
            return false;
        if ((weightsLayer->type == "Const") || (weightsLayer->type == "FakeQuantize")) {
            return true;
        }

        if (weightsLayer->type == "ScaleShift") {
            const std::vector<CNNLayerPtr> parents = NetworkHelper::getParents(*weightsLayer);
            if (parents.size() != 1ul) {
                return false;
            }

            return (parents[0]->type == "FakeQuantize") || (parents[0]->type == "Const");
        }

        return false;
    } else {
        return layer.blobs.find("weights") != layer.blobs.end();
    }
}

Blob::Ptr NetworkHelper::getWeights(
        const CNNLayer& layer,
        const bool roundQuantizedValues) {
    if (layer.insData.size() > 1) {
        CNNLayerPtr weightsLayer = NetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION << "Convolution weights const layer are absent";
        }

        if (weightsLayer->type == "Const") {
            NetworkHelper::checkConstWithBlobs(weightsLayer);
            return weightsLayer->blobs.find("custom")->second;
        } else if (weightsLayer->type == "FakeQuantize") {
            return NetworkHelper::quantizeWeights(*weightsLayer, roundQuantizedValues, Precision::UNSPECIFIED);
        } else if (weightsLayer->type == "ScaleShift") {
            const CNNLayerPtr parent = NetworkHelper::getParent(*weightsLayer);
            if (parent == nullptr)
                THROW_TRANSFORMATION_EXCEPTION << "Layer '" << weightsLayer->name << "' does not have parent";
            if (parent->type == "FakeQuantize") {
                return NetworkHelper::quantizeWeights(*parent, roundQuantizedValues, Precision::UNSPECIFIED);
            } else if (parent->type == "Const") {
                NetworkHelper::checkConstWithBlobs(parent);
                return NetworkHelper::getBlob(parent, "custom");
            } else {
                THROW_TRANSFORMATION_EXCEPTION <<
                    "Unexpected weights layer " << parent->type << " " << parent->name << " for " << layer.type << " " << layer.name;
            }
        } else {
            THROW_TRANSFORMATION_EXCEPTION << "Unexpected weights layer type " << weightsLayer->type;
        }
    } else {
        if (layer.blobs.find("weights") == layer.blobs.end()) {
            THROW_TRANSFORMATION_EXCEPTION << "Convolution weights are absent";
        }
        return layer.blobs.find("weights")->second;
    }
}

Blob::Ptr NetworkHelper::getBiases(const CNNLayer& layer) {
    if (layer.insData.size() > 1U) {
        if (layer.insData.size() > 2U) {
            CNNLayerPtr biasesLayer = NetworkHelper::getParent(layer, 2U);
            if (biasesLayer == nullptr) {
                return nullptr;
            }

            NetworkHelper::checkConstWithBlobs(biasesLayer);
            return biasesLayer->blobs.find("custom")->second;
        } else {
            return nullptr;
        }
    } else {
        const auto it = layer.blobs.find("biases");
        return (it != layer.blobs.end()) ? it->second : nullptr;
    }
}

#endif

std::shared_ptr<ngraph::opset1::Constant> NetworkHelper::quantizeWeights(
        std::shared_ptr<Node> quantize,
        const bool roundValues,
        const ngraph::element::Type precision) {
    std::cerr << "[ ERROR ] " << __FILE__ << ":" << __LINE__ << '\n';
    // FIXME: this is just a placeholder
    return std::make_shared<ngraph::opset1::Constant>(quantize->get_input_element_type(0), quantize->get_input_shape(0), 5);
#if 0 // TODO: LPT-TO-NGRAPH
    if (quantize.insData.size() != 5lu) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected inputs count: " << quantize.insData.size();
    }
    for (int i = 0; i < quantize.insData.size(); i++)
        if (quantize.insData[i].lock() == nullptr)
            THROW_TRANSFORMATION_EXCEPTION << "Invalid input data for layer '" << quantize.name << "' with index " << i;

    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "weights blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    const auto& sourceBlobTD = sourceBlob->getTensorDesc();
    const Precision blobPrecision = sourceBlobTD.getPrecision();

    auto targetBlobPrecision = precision == Precision::UNSPECIFIED ? blobPrecision : precision;
    if (targetBlobPrecision != Precision::FP32 && targetBlobPrecision != Precision::FP16 &&
        targetBlobPrecision != Precision::I8 && targetBlobPrecision != Precision::U8)
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected precision: " << precision;

    Blob::Ptr targetBlob = make_blob_with_precision(TensorDesc(targetBlobPrecision, sourceBlobTD.getDims(), sourceBlobTD.getLayout()));
    targetBlob->allocate();

    quantizeBlob(quantize, targetBlob, roundValues);

    return targetBlob;
#endif
}

#if 0 // TODO: LPT-TO-NGRAPH

int NetworkHelper::getConstParentBranchID(const CNNLayer& layer) {
    int constBranchID = -1;
    for (int i = 0; i < layer.insData.size(); i++) {
        bool allConst = true;

        const DataPtr insData = layer.insData[i].lock();
        if (insData == nullptr) {
            THROW_IE_LPT_EXCEPTION(layer) << "invalid input data with index " << i;
        }

        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_IE_LPT_EXCEPTION(layer) << "parent layer is absent";
        }

        if (!CaselessEq<std::string>()(parent->type, "FakeQuantize")) continue;
        for (const auto& p : parent->insData) {
            const DataPtr parentConstInsData = p.lock();
            if (parentConstInsData == nullptr) {
                THROW_IE_LPT_EXCEPTION(*parent) << "input data is absent";
            }
            const CNNLayerPtr parentConst = parentConstInsData->getCreatorLayer().lock();
            if (parentConst == nullptr) {
                THROW_IE_LPT_EXCEPTION(*parent) << "input layer is absent";
            }
            if (!CaselessEq<std::string>()(parentConst->type, "Const")) {
                allConst = false;
                break;
            }
        }
        if (allConst) {
            constBranchID = i;
            break;
        }
    }

    return constBranchID;
}

Precision NetworkHelper::getPrecisionParent(const CNNLayer& layer) {
    return getPrecisionParent(layer, 0ul, false);
}

Precision NetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex) {
    return getPrecisionParent(layer, parentIndex, true);
}

Precision NetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex) {
    const std::vector<CNNLayerPtr> parents = NetworkHelper::getParents(layer);
    if (parents.empty()) {
        THROW_TRANSFORMATION_EXCEPTION << "parents for layer " << layer.type << " '" << layer.name << "' are absent";
    }

    if (useParentIndex) {
        DataPtr parentOutData = getOutData(*parents[parentIndex], layer);
        if (parentOutData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION <<
                "parent layer " << parents[parentIndex]->type << " '" << parents[parentIndex]->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }
        return parentOutData->getTensorDesc().getPrecision();
    }

    Precision parentOutDataPrecision = Precision::UNSPECIFIED;
    for (CNNLayerPtr parent : parents) {
        DataPtr parentOutData = getOutData(*parent, layer);
        if (parentOutData == nullptr) {
            THROW_TRANSFORMATION_EXCEPTION <<
                "parent layer " << parent->type << " '" << parent->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }

        if (parentOutDataPrecision == Precision::UNSPECIFIED) {
            parentOutDataPrecision = parentOutData->getTensorDesc().getPrecision();
        } else if (parentOutDataPrecision != parentOutData->getTensorDesc().getPrecision()) {
            THROW_TRANSFORMATION_EXCEPTION <<
                "Parent layer " << parent->type << " '" << parent->name <<
                "' output port has unexpected precision " << parentOutData->getTensorDesc().getPrecision();
        }
    }

    return parentOutDataPrecision;
}

DataPtr NetworkHelper::getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer) {
    DataPtr parentOutData;
    for (DataPtr outData : parentLayer.outData) {
        const std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (auto childIt : inputTo) {
            if (childIt.second->name == childLayer.name) {
                parentOutData = outData;
                break;
            }
        }

        if (parentOutData != nullptr) {
            break;
        }
    }
    return parentOutData;
}

void NetworkHelper::quantizeBlob(const CNNLayer& quantize, Blob::Ptr& targetBlob, bool roundValues) {
    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
        THROW_TRANSFORMATION_EXCEPTION << "quantized blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    auto srcData = getFloatData(sourceBlob);
    const std::vector<size_t>& outDims = quantize.outData[0]->getDims();
    if (outDims.empty() || outDims.size() > 5lu) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected dimensions count " << outDims.size() << " for layer '" << quantize.name << "'";
    }

    // OIDHW
    const size_t OC = outDims[0];
    const size_t IC = outDims.size() > 1lu ? outDims[1] : 1;
    const size_t D  = outDims.size() > 4lu ? outDims[outDims.size() - 3] : 1;
    const size_t H  = outDims.size() > 2lu ? outDims[outDims.size() - 2] : 1;
    const size_t W  = outDims.size() > 3lu ? outDims[outDims.size() - 1] : 1;

    // Const layer blob shape (sourceBlob->getTensorDesc().getDims()) can be different from output port shape
    // CVS-27850: [IE COMMON] Align Const layer blob shape with output port shape
    if (sourceBlob->size() != OC * IC * D * H * W) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected weights size for layer '" << quantize.name << "'";
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(quantize);

    const bool isInputLowBroadcasted = quantizationDetails.inputLowValues.size() != OC;
    if ((quantizationDetails.inputLowValues.size() != 1) && (quantizationDetails.inputLowValues.size() != OC)) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected input low values count " << quantizationDetails.inputLowValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isInputHighBroadcasted = quantizationDetails.inputHighValues.size() != OC;
    if ((quantizationDetails.inputHighValues.size() != 1) && (quantizationDetails.inputHighValues.size() != OC)) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected input high values count " << quantizationDetails.inputHighValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isOutputLowBroadcasted = quantizationDetails.outputLowValues.size() != OC;
    if ((quantizationDetails.outputLowValues.size() != 1) && (quantizationDetails.outputLowValues.size() != OC)) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputLowValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isOutputHighBroadcasted = quantizationDetails.outputHighValues.size() != OC;
    if ((quantizationDetails.outputHighValues.size() != 1) && (quantizationDetails.outputHighValues.size() != OC)) {
        THROW_TRANSFORMATION_EXCEPTION << "Unexpected output high values count " << quantizationDetails.outputHighValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    auto levels_1 = static_cast<float>(quantize.GetParamAsUInt("levels")) - 1.f;

    const size_t DHW = D * H * W;
    const size_t IDHW = IC * DHW;

    std::vector<float> dstBuffer(targetBlob->size());

    auto srcPtr = srcData.get();
    auto dstPtr = &dstBuffer[0];

    parallel_for4d(OC, IC, D, H, [&](size_t oc, size_t ic, size_t d, size_t h) {
        const float inputLow = quantizationDetails.inputLowValues[isInputLowBroadcasted ? 0 : oc];
        const float inputHigh = quantizationDetails.inputHighValues[isInputHighBroadcasted ? 0 : oc];
        const float outputLow = quantizationDetails.outputLowValues[isOutputLowBroadcasted ? 0 : oc];
        const float outputHigh = quantizationDetails.outputHighValues[isOutputHighBroadcasted ? 0 : oc];

        for (size_t w = 0; w < W; w++) {
            const size_t idx = oc * IDHW + ic * DHW + d * H * W + h * W + w;

            if (srcPtr[idx] <= inputLow) {
                dstPtr[idx] = roundValues ? std::roundf(outputLow) : outputLow;
            } else if (srcPtr[idx] > inputHigh) {
                dstPtr[idx] = roundValues ? std::roundf(outputHigh) : outputHigh;
            } else {
                const float value = std::roundf((srcPtr[idx] - inputLow) / (inputHigh - inputLow) * levels_1) /
                                    levels_1 * (outputHigh - outputLow) + outputLow;
                dstPtr[idx] = roundValues ? std::roundf(value) : value;
            }
        }
    });

    fillBlobByFP32(targetBlob, dstPtr);
}

#endif

std::shared_ptr<opset1::Add> decomposeMultiplyAdd(std::shared_ptr<op::MultiplyAdd> multiplyAdd) {
    using namespace std;
    using namespace ngraph::op;
    // FIXME: need to modify data_type on output to be aligned with MultiplyAdd output
    // it is fundamental limitation of TypeRelaxed approach when constructing new graphs
    //NetworkHelper::setOutDataPrecision(multiplyAdd->input_value(0).get_node_shared_ptr(), multiplyAdd->get_output_element_type(0));
    AutoReplaceInputTypes<Node> auto_type(*multiplyAdd, multiplyAdd->get_output_element_type(0));
    auto multiply = make_shared<TypeRelaxed<opset1::Multiply>>(
            opset1::Multiply(multiplyAdd->input_value(0), multiplyAdd->input_value(1)), multiplyAdd->get_output_element_type(0));
    auto add = make_shared<opset1::Add>(multiply, multiplyAdd->input_value(2));
    copy_runtime_info(multiplyAdd, {multiply, add});
    add->set_friendly_name(multiplyAdd->get_friendly_name());
    replace_node(multiplyAdd, add);
    return add;
}

std::shared_ptr<opset1::Multiply> swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply) {
    // Multiply --> Add(addAfterMultiply)  ==>  Add(new) --> Multiply(new)
    // That means x*a + b ==> (x + b/a)*a; tries to fold b/a
    auto x = addAfterMultiply->input_value(0).get_node()->input_value(0);
    auto a = addAfterMultiply->input_value(0).get_node()->input_value(1);
    auto b = addAfterMultiply->input_value(1);
    auto bDivA = std::make_shared<opset1::Divide>(b, a);
    OutputVector foldedTerm;
    if (bDivA->constant_fold(foldedTerm)) {
        assert(foldedTerm.size() == 1);
        auto addTerm = as_type_ptr<opset1::Constant>(foldedTerm[0].get_node_shared_ptr());
        // TODO: is it useful to optimize here?
#if 0
        if (isScalarLike(addTerm) && addTerm->cast_vector<float>()[0] == 0) {
            foldedTerm.clear();
        } else {
#endif
            replace_node(bDivA, foldedTerm);
#if 0
        }
#endif
    } else {
        foldedTerm = {bDivA->output(0)};
    }
    op::AutoReplaceInputTypes<Node> auto_type(*addAfterMultiply->input_value(0).get_node(), addAfterMultiply->get_output_element_type(0));
    Output<Node> newMultiplyInput;
    if (!foldedTerm.empty()) {
        auto newAdd = std::make_shared<op::TypeRelaxed<opset1::Add>>(opset1::Add(x, foldedTerm[0]),
                                                                     addAfterMultiply->get_output_element_type(0));
        newMultiplyInput = newAdd->output(0);
    } else {
        newMultiplyInput = x;
    }
    auto newMultiply = std::make_shared<opset1::Multiply>(newMultiplyInput, a);
    replace_node(addAfterMultiply, newMultiply);
    return newMultiply;
}

bool isScalarLike(std::shared_ptr<opset1::Constant> constant) {
#if 1
    return constant->get_all_data_elements_bitwise_identical();
#else
    // FIXME: work for floats only
    const auto scalesBuffer = constant->cast_vector<float>();
    size_t scalesBufferSize = shape_size(constant->get_output_shape(0));

    for (size_t i = 1ul; i < scalesBufferSize; ++i) {
        if (scalesBuffer[i - 1ul] != scalesBuffer[i]) {
            return false;
        }
    }
    return true;
#endif
}

std::shared_ptr<opset1::Constant> distillToScalar(std::shared_ptr<opset1::Constant> constant) {
    assert(isScalarLike(constant));
    return std::make_shared<opset1::Constant>(constant->get_element_type(), Shape{}, constant->get_data_ptr());
}

std::shared_ptr<Node> getConstantInput(std::shared_ptr<Node> node) {
    std::shared_ptr<Node> constant1 = as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr());
    if (!constant1) {
        constant1 = as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr());
    }
    return constant1;
}


std::shared_ptr<Node> optimizeMultipliesAfter(std::shared_ptr<Node> multiply) {
    multiply = as_type_ptr<opset1::Multiply>(multiply);

    if (multiply && multiply->output(0).get_target_inputs().size() == 1) {
        auto constant1 = getConstantInput(multiply);
        if (!constant1 || constant1->output(0).get_target_inputs().size() != 1) {
            return multiply;
        }
        auto nextMultiplyInput = *multiply->output(0).get_target_inputs().begin();
        auto nextMultiply = as_type_ptr<opset1::Multiply>(nextMultiplyInput.get_node()->shared_from_this());
        if (nextMultiply) {
            auto constant2 = getConstantInput(nextMultiply);
            if (!constant2 || constant2->output(0).get_target_inputs().size() != 1) {
                return multiply;
            }

            auto newConst = fold<opset1::Multiply>(constant1, constant2);
            auto newMultiply =
                    std::make_shared<opset1::Multiply>(
                            multiply->input_value(1 - constant1->output(0).get_target_inputs().begin()->get_index()),
                            newConst->output(0));
            replace_node(nextMultiply, newMultiply);
            return newMultiply;
        }
    }

    return nullptr;
}

std::shared_ptr<opset1::Constant> roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance) {
    auto constant = as_type_ptr<opset1::Constant>(node);
    assert(constant);
    auto values = constant->cast_vector<float>();

    auto castedConstant = as_type_ptr<opset1::Constant>(fold<opset1::Convert>(constant, target_type));
    auto castedValues = castedConstant->cast_vector<float>();

    // TODO: implement with constant folding when ReduceAnd constant folding is ready
    if (std::equal(values.begin(), values.end(), castedValues.begin(), [tolerance](float a, float b) { return fabs(a - b) < tolerance; })) {
        return castedConstant;
    } else {
        return constant;
    }
}

// Decompose FakeQuantize to FakeQuantize with output integer limits (quantize), dequatized MultiplyAdd
// To align types the resulting sequence is FakeQuantize -> Convert -> Convert -> MultiplyAdd
std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(std::shared_ptr<opset1::FakeQuantize> fq,
                                            element::Type precision,
                                            float min,
                                            float max) {
    using std::make_shared;

    // Now calculate scales and shifts according to given shapes -- all operations in ngraph
    auto newMin = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, min);
    auto newMax = make_shared<opset1::Constant>(fq->get_output_element_type(0), Shape{}, max);

    auto outputLow = fq->input_value(3);
    auto outputHigh = fq->input_value(4);

    std::shared_ptr<Node> scale;
    std::shared_ptr<Node> shift;

    if (precision.is_signed()) {
        // I8
        scale = fold<opset1::Divide>(
                fold<opset1::Subtract>(outputHigh, outputLow),
                fold<opset1::Subtract>(newMax, newMin));

        auto actualLowPartQuantValue = fold<opset1::Abs>(fold<opset1::Divide>(outputLow, newMin));
        auto actualHighPartQuantValue = fold<opset1::Abs>(fold<opset1::Divide>(outputHigh, newMax));

        shift = fold<opset1::Select>(
                fold<opset1::Less>(actualLowPartQuantValue, actualHighPartQuantValue),
                fold<opset1::Subtract>(outputLow, fold<opset1::Multiply>(newMin, scale)),
                fold<opset1::Subtract>(outputHigh, fold<opset1::Multiply>(newMax, scale)));

    } else {
        // U8
        scale = fold<opset1::Divide>(
                fold<opset1::Subtract>(outputHigh, outputLow),
                fold<opset1::Subtract>(newMax, newMin));

        // TODO: here should be a check for zero point; I've removed it as it always true
        // if it is really required
        shift = outputLow.get_node_shared_ptr();
    }

    // Build a substitution sub-graph:
    auto newFQ = fold_fake_quantize<opset1::FakeQuantize>(
            fq->input_value(0),
            fq->input_value(1),
            fq->input_value(2),
            newMin->output(0),
            newMax->output(0),
            fq->get_levels(),
            fq->get_auto_broadcast());

    auto dequantize = make_shared<ngraph::op::MultiplyAdd>(
            make_shared<opset1::Convert>(
                    fold<opset1::Convert>(newFQ, precision),
                    fq->get_output_element_type(0)), scale, shift);
    replace_node(fq, dequantize);
    // Make type-relaxed node

#if 0
    // FIXME: is it needed?
    if (fabs(dequantizationScale) < minQuantizationScale) {
        dequantizationScales[channel] = minQuantizationScale;
        denormalOutputValuesWasUpdated = true;
    } else if (fabs(dequantizationScale) > maxQuantizationScale) {
        dequantizationScales[channel] = dequantizationScale > 0.f ? maxQuantizationScale : -maxQuantizationScale;
        denormalOutputValuesWasUpdated = true;
    } else {
        dequantizationScales[channel] = dequantizationScale;
    }
#endif

    return std::make_tuple(newFQ, dequantize);
}

std::shared_ptr<Node> optimizeAdd(std::shared_ptr<opset1::Add> add) {
    auto convertOnAdd = add->input_value(0).get_node_shared_ptr();
    // TODO: replace assert to condition and omit conversion part if there is no convert
    // TODO: also check convertInputType to understand if we really want to propagate type
    assert(as_type_ptr<opset1::Convert>(convertOnAdd));
    auto convertInputType = convertOnAdd->get_input_element_type(0);
    auto data = convertOnAdd->input_value(0);
    auto shift = add->input_value(1).get_node_shared_ptr();
    auto roundedShift = roundWithTolerance(shift, convertInputType);
    std::shared_ptr<Node> replacement;
    if (roundedShift->get_element_type() == convertInputType) {
        // Propagate convertInputType down
        replacement = std::make_shared<opset1::Add>(data, roundedShift);
        replace_node(add, replacement);
    } else {
        // Try to represent it as data - (-b)
        roundedShift = roundWithTolerance(fold<opset1::Negative>(shift), convertInputType);
        if (roundedShift->get_element_type() == convertInputType) {
            // Assuming Subtract will go out of representable set of values for target type
            // So keep the original data type (likely not integer)
            replacement = std::make_shared<op::TypeRelaxed<opset1::Subtract>>(
                    opset1::Subtract(data, roundedShift),
                    convertOnAdd->get_output_element_type(0));
            replace_node(add, replacement);
        }
    }

    // We lose the tail conversion here; not needed if the next node is a TypeRelaxed
    // TODO: check cases when Convert should be preserved

    // Try to optimize Add out if constant is zero
    if (isScalarLike(roundedShift)) {
        auto scalar = distillToScalar(roundedShift);
        if (op::util::constantIsEqualTo(scalar, 0)) {
            replace_node(replacement, replacement->input_value(0).get_node_shared_ptr());
            replacement = nullptr;
        }
    }

    return replacement;
}



}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
