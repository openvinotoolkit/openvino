// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/transformer.hpp"
#include "low_precision_transformations/network_helper.hpp"
#include "itt.hpp"

#include <ie_common.h>

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <legacy/cnn_network_impl.hpp>
#include <legacy/ie_util_internal.hpp>

#include "low_precision_transformations/activation.hpp"
#include "low_precision_transformations/concat_multi_channels.hpp"
#include "low_precision_transformations/const.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/depth_to_space.hpp"
#include "low_precision_transformations/fake_quantize.hpp"
#include "low_precision_transformations/fully_connected.hpp"
#include "low_precision_transformations/fuse_fake_quantize_and_scale_shift.hpp"
#include "low_precision_transformations/gemm.hpp"
#include "low_precision_transformations/mvn.hpp"
#include "low_precision_transformations/permute.hpp"
#include "low_precision_transformations/pooling.hpp"
#include "low_precision_transformations/resample.hpp"
#include "low_precision_transformations/power.hpp"
#include "low_precision_transformations/reshape.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"
#include "low_precision_transformations/squeeze.hpp"
#include "low_precision_transformations/eltwise.hpp"
#include "low_precision_transformations/normalize.hpp"

// uncomment to display precision info during low precision transformations
// #define DISPLAY_PECISION

using namespace InferenceEngine;
using namespace InferenceEngine::details;

LowPrecisionTransformations::LowPrecisionTransformations(
    const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
    const std::map<std::string, LayerTransformationPtr>& transformations,
    const std::map<std::string, LayerTransformationPtr>& cleanupTransformations) :
    branchSpecificTransformations(branchSpecificTransformations),
    transformations(transformations),
    cleanupTransformations(cleanupTransformations) {}

void LowPrecisionTransformations::setUpdatePrecisions(const bool updatePrecisions) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setUpdatePrecisions(updatePrecisions);
    }
}

void LowPrecisionTransformations::setQuantizeOutputs(const bool quantizeOutputs) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizeOutputs(quantizeOutputs);
    }
}

void LowPrecisionTransformations::setWeightsToConst(const bool weightsToConst) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setWeightsToConst(weightsToConst);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setWeightsToConst(weightsToConst);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnActivations(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnActivations(quantizedTensorAlignmentOnActivations);
    }
}

void LowPrecisionTransformations::setQuantizedTensorAlignmentOnWeights(
    const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights) {
    for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
    for (auto it = transformations.begin(); it != transformations.end(); ++it) {
        it->second->setQuantizedTensorAlignmentOnWeights(quantizedTensorAlignmentOnWeights);
    }
}

LowPrecisionTransformations& LowPrecisionTransformations::remove(const std::string& layerType) {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    removeBranchSpecificTransformations(type);
    removeTransformations(type);
    removeCleanupTransformations(type);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeBranchSpecificTransformations(const std::string& layerType) {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    branchSpecificTransformations.erase(type);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeTransformations(const std::string& layerType) {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    transformations.erase(type);
    return *this;
}

LowPrecisionTransformations& LowPrecisionTransformations::removeCleanupTransformations(const std::string& layerType) {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    cleanupTransformations.erase(type);
    return *this;
}

LayerTransformationPtr LowPrecisionTransformations::find(const std::string& layerType) const {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    auto it = branchSpecificTransformations.find(type);
    if (it != branchSpecificTransformations.end()) {
        return it->second;
    }

    it = transformations.find(type);
    if (it != transformations.end()) {
        return it->second;
    }

    it = cleanupTransformations.find(type);
    if (it != cleanupTransformations.end()) {
        return it->second;
    }

    return nullptr;
}

void LowPrecisionTransformations::setParamsManager(IParamsManager* paramsManager) noexcept {
    setParamsManager(paramsManager, branchSpecificTransformations);
    setParamsManager(paramsManager, transformations);
    setParamsManager(paramsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept {
    setLayerTransformationsManager(layerTransformationsManager, branchSpecificTransformations);
    setLayerTransformationsManager(layerTransformationsManager, transformations);
    setLayerTransformationsManager(layerTransformationsManager, cleanupTransformations);
}

void LowPrecisionTransformations::setParamsManager(
    IParamsManager* paramsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setParamsManager(paramsManager);
    }
}

void LowPrecisionTransformations::setLayerTransformationsManager(
    ILayerTransformationsManager* layerTransformationsManager,
    std::map<std::string, LayerTransformationPtr>& transformations) noexcept {
    for (auto it : transformations) {
        it.second->setLayerTransformationsManager(layerTransformationsManager);
    }
}

LowPrecisionTransformations LowPrecisionTransformer::getAllTransformations(const LayerTransformation::Params& params) {
    return LowPrecisionTransformations(
        std::map<std::string, LayerTransformationPtr>({
            { "concat", LayerTransformationPtr(new ConcatMultiChannelsTransformation(params))}
        }),
        std::map<std::string, LayerTransformationPtr>({
            { "convolution", LayerTransformationPtr(new ConvolutionTransformation(params)) },
            { "pooling", LayerTransformationPtr(new PoolingTransformation(params)) },
            { "fakequantize", LayerTransformationPtr(new FakeQuantizeTransformation(params)) },
            { "reshape", LayerTransformationPtr(new ReshapeTransformation(params)) },
            { "fullyconnected", LayerTransformationPtr(new FullyConnectedTransformation(params)) },
            { "gemm", LayerTransformationPtr(new GemmTransformation(params)) },
            { "permute", LayerTransformationPtr(new PermuteTransformation(params)) },
            { "squeeze", LayerTransformationPtr(new SqueezeTransformation(params)) },
            { "relu", LayerTransformationPtr(new ActivationTransformation(params)) },
            { "mvn", LayerTransformationPtr(new MvnTransformation(params)) },
            { "eltwise", LayerTransformationPtr(new EltwiseTransformation(params)) },
            { "resample", LayerTransformationPtr(new ResampleTransformation(params)) },
            { "power", LayerTransformationPtr(new PowerTransformation(params)) },
            { "depthtospace", LayerTransformationPtr(new DepthToSpaceTransformation(params)) },
            { "normalize", LayerTransformationPtr(new NormalizeTransformation(params)) }
        }),
        std::map<std::string, LayerTransformationPtr>({
            { "fakequantize", LayerTransformationPtr(new FuseFakeQuantizeAndScaleShiftTransformation(params)) },
            { "scaleshift", LayerTransformationPtr(new ScaleShiftToConvolutionTransformation(params)) },
        }));
}

LowPrecisionTransformer::LowPrecisionTransformer(): transformations(LowPrecisionTransformer::getAllTransformations()) {}

LowPrecisionTransformer::LowPrecisionTransformer(const LowPrecisionTransformations& transformations)
    : transformations(transformations) {}

void LowPrecisionTransformer::renameLayersByType(const std::vector<CNNLayerPtr>& layers, const std::string& type) {
    size_t number = 1;
    for (size_t i = 0; i < layers.size(); ++i) {
        const CNNLayerPtr layer = layers[i];
        if (layer->type != type) {
            continue;
        }

        layer->name = layer->type + std::to_string(number);
        ++number;
    }
}

void LowPrecisionTransformer::rename(ICNNNetwork& network) const {
    TransformationContext context(network);

    const std::unordered_set<std::string> standaloneLayerTypes = {"Convolution", "Concat",  "Eltwise",
                                                                  "Reshape",     "Pooling", "Clamp"};
    for (const std::string& standaloneLayerType : standaloneLayerTypes) {
        renameLayersByType(context.getLayers(), standaloneLayerType);
    }

    size_t fakeQuantizeNumber = 1;
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer->type != "FakeQuantize") {
            continue;
        }

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
        if ((children.size() == 1) && (children[0]->type == "Convolution")) {
            const std::string postfix = CNNNetworkHelper::getIndex(*layer) == 0 ? "data" : "weights";
            layer->name = children[0]->name + "_FakeQuantize_" + postfix;
        } else {
            layer->name = layer->type + std::to_string(fakeQuantizeNumber);
            ++fakeQuantizeNumber;
        }
    }

    size_t otherNumber = 1;
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
        std::string name;
        const CNNLayerPtr layer = context.getLayers()[i];
        if ((standaloneLayerTypes.find(layer->type) != standaloneLayerTypes.end()) || (layer->type == "FakeQuantize")) {
            continue;
        }

        if (layer->type == "Const") {
            const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
            if (children.size() == 1) {
                if (children[0]->type == "Convolution") {
                    const std::string postfix = CNNNetworkHelper::getIndex(*layer) == 1 ? "weights" : "biases";
                    name = children[0]->name + "_Const_" + postfix;
                } else if (children[0]->type == "FakeQuantize") {
                    name = children[0]->name + "_Const_" + std::to_string(CNNNetworkHelper::getIndex(*layer));
                }
            }
        }

        if (name.empty()) {
            name = layer->type + std::to_string(otherNumber);
            ++otherNumber;
        }

        layer->name = name;
    }
}

void LowPrecisionTransformer::transform(ICNNNetwork& network) {
    OV_ITT_SCOPED_TASK(itt::domains::LPT, "LowPrecisionTransformer::transform");

#ifdef LPT_ORIGINAL_MODEL_PATH
    ResponseDesc originalModelResponse;
    network.serialize(
        std::string(LPT_ORIGINAL_MODEL_PATH) + ".xml",
        std::string(LPT_ORIGINAL_MODEL_PATH) + ".bin",
        &originalModelResponse);
    if (originalModelResponse.msg[0] != '\0') {
        THROW_IE_EXCEPTION << "LowPrecisionTransformer::transform: " << LPT_ORIGINAL_MODEL_PATH << ": " << originalModelResponse.msg;
    }
#endif
    auto it = details::CNNNetworkIterator(&network);
    auto end = details::CNNNetworkIterator();
    bool fqFound = false;
    bool allFQareUnsupported = true;
    while (it != end) {
        if (CaselessEq<std::string>()((*it)->type, "FakeQuantize")) {
            fqFound = true;
            if (QuantizationDetails::isSupportedLevel((*it)->GetParamAsUInt("levels"))) {
                allFQareUnsupported = false;
                break;
            }
        }
        it++;
    }
    // If network does not have FakeQuantize layers
    // or all found FQ layers are binary - do nothing and return
    if (!fqFound || allFQareUnsupported) return;

    transformations.setParamsManager(this);
    transformations.setLayerTransformationsManager(this);

    TransformationContext context(network);

    // TODO: branch specific transformations execution
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
            continue;
        }

        std::string type = layer->type;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        const auto it = transformations.branchSpecificTransformations.find(type);
        if (it == transformations.branchSpecificTransformations.end()) {
            continue;
        }
        it->second->transform(context, *layer);
    }

    // Step #1: FakeQuantize layer transformation execution
    LayerTransformationPtr fqTransformation = transformations.find("FakeQuantize");
    if (fqTransformation == nullptr) {
        THROW_IE_EXCEPTION << "FakeQuantize transformation was not found";
    }
    for (size_t i = 0lu; i < context.getLayers().size(); ++i) {
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
            continue;
        }

        if (CaselessEq<std::string>()(layer->type, "FakeQuantize")) {
            fqTransformation->transform(context, *layer);
        }
    }

    // Step #2: layer transformations execution
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
            continue;
        }

        bool transformed;

        std::string type = layer->type;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        const auto it = transformations.transformations.find(type);
        if (it != transformations.transformations.end()) {
            it->second->transform(context, *layer);
            transformed = true;
        }

#ifdef DISPLAY_PECISION
        CNNLayerPtr transformedLayer = CNNNetworkHelper::getLayer(context.network, layer->name);
        if (transformedLayer == nullptr) {
            if (layer->type == "FakeQuantize") {
                std::cout << "Layer " << layer->name << ": " << QuantizationDetails::getDetails(*layer) << std::endl;
            }

            std::cout << "Layer was " << (transformed ? "transformed: " : "skipped: ") << layer->type << ", "
                      << layer->name << ": [REMOVED]" << std::endl;
        } else {
            if (transformedLayer->type == "FakeQuantize") {
                std::cout << "Layer " << transformedLayer->name << ": "
                          << QuantizationDetails::getDetails(*transformedLayer) << std::endl;
            }

            std::cout << "Layer was " << (transformed ? "transformed: " : "skipped: ") << transformedLayer->type << ", "
                      << transformedLayer->name << ", output layer precision: "
                      << ((transformedLayer->outData.size() != 0) ? transformedLayer->outData[0]->getPrecision()
                                                                  : Precision::UNSPECIFIED)
                      << std::endl;
        }

#endif
    }

    // Step #3: cleanup transformations execution
    for (size_t i = 0; i < context.getLayers().size(); ++i) {
        const CNNLayerPtr layer = context.getLayers()[i];
        if (layer == nullptr) {
            continue;
        }

        std::string type = layer->type;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);
        const auto it = transformations.cleanupTransformations.find(type);
        if (it != transformations.cleanupTransformations.end()) {
            it->second->transform(context, *layer);
        }
    }

#ifdef LPT_TRANSFORMED_MODEL_PATH
    ResponseDesc transformedModelResponse;
    network.serialize(
        std::string(LPT_TRANSFORMED_MODEL_PATH) + ".xml",
        std::string(LPT_TRANSFORMED_MODEL_PATH) + ".bin",
        &transformedModelResponse);
    if (transformedModelResponse.msg[0] != '\0') {
        THROW_IE_EXCEPTION << "LowPrecisionTransformer::transform: " << LPT_TRANSFORMED_MODEL_PATH << ": " << transformedModelResponse.msg;
    }
#endif
}

std::vector<Precision> LowPrecisionTransformer::getPrecisionsOnActivations(const std::string& layerType) const noexcept {
    std::string type = layerType;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    const LayerTransformationPtr transformation = transformations.find(type);
    if (transformation == nullptr) {
        return std::vector<Precision>();
    }
    return transformation->getPrecisionsOnActivations();
}

bool LowPrecisionTransformer::isQuantized(const CNNLayer& layer) const noexcept {
    std::string type = layer.type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    const LayerTransformationPtr transformation = transformations.find(type);
    if (transformation == nullptr) {
        return false;
    }
    return transformation->isQuantized(layer);
}

bool LowPrecisionTransformer::isPrecisionPreserved(const CNNLayer& layer) const noexcept {
    std::string type = layer.type;
    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    const LayerTransformationPtr transformation = transformations.find(type);
    if (transformation == nullptr) {
        return false;
    }
    return transformation->isPrecisionPreserved(layer);
}
