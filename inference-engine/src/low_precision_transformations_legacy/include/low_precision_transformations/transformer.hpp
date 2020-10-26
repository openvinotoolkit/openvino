// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "layer_transformation.hpp"
#include "iparams_manager.hpp"
#include "ilayer_transformations_manager.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(LowPrecisionTransformations) {
public:
    LowPrecisionTransformations(
        const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
        const std::map<std::string, LayerTransformationPtr>& transformations,
        const std::map<std::string, LayerTransformationPtr>& cleanupTransformations);

    void setUpdatePrecisions(const bool updatePrecisions);
    void setQuantizeOutputs(const bool quantizeOutputs);
    void setWeightsToConst(const bool weightsToConst);
    void setQuantizedTensorAlignmentOnActivations(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations);
    void setQuantizedTensorAlignmentOnWeights(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights);
    LowPrecisionTransformations& remove(const std::string& layerType);
    LowPrecisionTransformations& removeBranchSpecificTransformations(const std::string& layerType);
    LowPrecisionTransformations& removeTransformations(const std::string& layerType);
    LowPrecisionTransformations& removeCleanupTransformations(const std::string& layerType);

    template <class T>
    LowPrecisionTransformations& addBranchSpecific(const LayerTransformation::Params& params, const std::string& layerType) {
        std::string type = layerType;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);

        const auto it = branchSpecificTransformations.find(type);
        if (it != branchSpecificTransformations.end()) {
            branchSpecificTransformations.erase(it);
        }

        branchSpecificTransformations.emplace(type, std::make_shared<T>(params));
        return *this;
    }

    template <class T>
    LowPrecisionTransformations& add(const LayerTransformation::Params& params, const std::string& layerType) {
        std::string type = layerType;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);

        const auto it = transformations.find(type);
        if (it != transformations.end()) {
            transformations.erase(it);
        }

        transformations.emplace(type, std::make_shared<T>(params));
        return *this;
    }

    template <class T>
    LowPrecisionTransformations& addCleanup(const LayerTransformation::Params& params, const std::string& layerType) {
        std::string type = layerType;
        std::transform(type.begin(), type.end(), type.begin(), ::tolower);

        const auto it = cleanupTransformations.find(type);
        if (it != cleanupTransformations.end()) {
            cleanupTransformations.erase(it);
        }

        cleanupTransformations.emplace(type, std::make_shared<T>(params));
        return *this;
    }

    LayerTransformationPtr find(const std::string& layerType) const;

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    std::map<std::string, LayerTransformationPtr> branchSpecificTransformations;
    std::map<std::string, LayerTransformationPtr> transformations;
    std::map<std::string, LayerTransformationPtr> cleanupTransformations;

private:
    static void setParamsManager(IParamsManager* paramsManager, std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
    static void setLayerTransformationsManager(
        ILayerTransformationsManager* layerTransformationsManager,
        std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
};

/**
 * @brief low precision transformation component.
  */
class INFERENCE_ENGINE_API_CLASS(LowPrecisionTransformer) : public IParamsManager, ILayerTransformationsManager {
public:
    static LowPrecisionTransformations getAllTransformations(const LayerTransformation::Params& params = LayerTransformation::Params());

    LowPrecisionTransformer();
    LowPrecisionTransformer(const LowPrecisionTransformations& transformations);
    void transform(ICNNNetwork& network);
    void rename(ICNNNetwork& network) const;

    // IParamsManager interface implementation
    std::vector<Precision> getPrecisionsOnActivations(const std::string& layerName) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(const CNNLayer& layer) const noexcept override;
    bool isPrecisionPreserved(const CNNLayer& layer) const noexcept override;

private:
    static void renameLayersByType(const std::vector<CNNLayerPtr>& layers, const std::string& type);
    LowPrecisionTransformations transformations;
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
