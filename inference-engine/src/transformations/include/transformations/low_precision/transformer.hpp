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

namespace ngraph {
namespace pass {
namespace low_precision {


class TRANSFORMATIONS_API LowPrecisionTransformations {
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
    LowPrecisionTransformations& remove(const std::string& layerName);
    LowPrecisionTransformations& removeBranchSpecificTransformations(const std::string& layerName);
    LowPrecisionTransformations& removeTransformations(const std::string& layerName);
    LowPrecisionTransformations& removeCleanupTransformations(const std::string& layerName);

    template <class T>
    LowPrecisionTransformations& addBranchSpecific(const LayerTransformation::Params& params, const std::string& transformationName) {
        const auto it = branchSpecificTransformations.find(transformationName);
        if (it != branchSpecificTransformations.end()) {
            branchSpecificTransformations.erase(it);
        }

        branchSpecificTransformations.emplace(transformationName, std::make_shared<T>(params));
        return *this;
    }

    template <class T>
    LowPrecisionTransformations& add(const LayerTransformation::Params& params, const std::string& transformationName) {
        const auto it = transformations.find(transformationName);
        if (it != transformations.end()) {
            transformations.erase(it);
        }

        transformations.emplace(transformationName, std::make_shared<T>(params));
        return *this;
    }

    template <class T>
    LowPrecisionTransformations& addCleanup(const LayerTransformation::Params& params, const std::string& transformationName) {
        const auto it = cleanupTransformations.find(transformationName);
        if (it != cleanupTransformations.end()) {
            cleanupTransformations.erase(it);
        }

        cleanupTransformations.emplace(transformationName, std::make_shared<T>(params));
        return *this;
    }

    LayerTransformationPtr find(const std::string& transformationName) const;

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    // Key is not a layer type, but just a name of transformation
    // Layer type (or a pattern) is defined by transformation itself as an ngraph matcher
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
class TRANSFORMATIONS_API LowPrecisionTransformer : public IParamsManager, ILayerTransformationsManager {
public:
    static LowPrecisionTransformations getAllTransformations(const LayerTransformation::Params& params = LayerTransformation::Params());

    LowPrecisionTransformer();
    LowPrecisionTransformer(const LowPrecisionTransformations& transformations);
    void transform(std::shared_ptr<Function> network);
#if 0 // TODO LPT-TO-NGRAPH
    void rename(std::shared_ptr<Function> network) const;
#endif

    // IParamsManager interface implementation
    std::vector<element::Type> getPrecisionsOnActivations(const std::string& layerType) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(std::shared_ptr<Node> layer) const noexcept override;
    bool isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept override;

private:
#if 0 // TODO LPT-TO-NGRAPH
    static void renameLayersByType(const std::vector<std::shared_ptr<Node>>& layers, const NodeTypeInfo& layerType);
#endif
    LowPrecisionTransformations transformations;

    void registerAllMatchers(
        std::map<std::string, LayerTransformationPtr> transformations,
        GraphRewrite& pass,
        TransformationContext& context);
};

}// namespace low_precision
}// namespace pass
}// namespace ngraph