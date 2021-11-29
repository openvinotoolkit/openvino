// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "layer_transformation.hpp"
#include "iparams_manager.hpp"
#include "ilayer_transformations_manager.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

struct StandaloneCleanup {
    std::string typeName;
    std::string typeId;
    LayerTransformationPtr transformation;
};

class TRANSFORMATIONS_API LowPrecisionTransformations {
public:
    LowPrecisionTransformations() {}
    LowPrecisionTransformations(
        const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
        const std::map<std::string, LayerTransformationPtr>& decompositionTransformations,
        const std::map<std::string, LayerTransformationPtr>& transformations,
        const std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& cleanupTransformations,
        const std::vector<StandaloneCleanup>& standaloneCleanupTransformations);

    void setUpdatePrecisions(const bool updatePrecisions);
    void setQuantizedTensorAlignmentOnActivations(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations);
    void setQuantizedTensorAlignmentOnWeights(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights);

    /**
     * Remove branch specific transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& removeBranchSpecific() {
        const std::string operationType = getType<Operation>();
        const std::string transformationType = typeid(Transformation).name();

        for (auto it = branchSpecificTransformations.begin(); it != branchSpecificTransformations.end(); ++it) {
            const auto& tranformationPtr = *it->second;
            if ((it->first == operationType) && (typeid(tranformationPtr).name() == transformationType)) {
                branchSpecificTransformations.erase(it);
                break;
            }
        }
        return *this;
    }

    /**
     * Remove transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& remove() {
        const std::string operationType = getType<Operation>();
        const std::string transformationType = typeid(Transformation).name();

        for (auto it = transformations.begin(); it != transformations.end(); ++it) {
            const auto& tranformationPtr = *it->second;
            if ((it->first == operationType) && (typeid(tranformationPtr).name() == transformationType)) {
                transformations.erase(it);
                break;
            }
        }
        return *this;
    }

    /**
     * Remove cleanup transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& removeCleanup() {
        const std::string operationType = getType<Operation>();
        const std::string transformationType = typeid(Transformation).name();

        const auto it = cleanupTransformations.find(operationType);
        if (it != cleanupTransformations.end()) {
            const auto it1 = std::find_if(it->second.begin(), it->second.end(),
                [&](const std::pair<std::string, LayerTransformationPtr>& transformation) {
                    return transformation.first == transformationType;
                });
            if (it1 != it->second.end()) {
                it->second.erase(it1);
                if (it->second.empty()) {
                    cleanupTransformations.erase(it);
                }
            }
        }
        return *this;
    }

    /**
     * Remove standalone cleanup transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& removeStandaloneCleanup() {
        const std::string operationType = getType<Operation>();
        const std::string transformationType = typeid(Transformation).name();

        for (auto it = standaloneCleanupTransformations.begin(); it != standaloneCleanupTransformations.end(); ++it) {
            const auto& standaloneCleanup = *it;
            if ((operationType == standaloneCleanup.typeName) && (transformationType == standaloneCleanup.typeId)) {
                standaloneCleanupTransformations.erase(it);
                break;
            }
        }
        return *this;
    }

    template <class Transformation, class Operation>
    LowPrecisionTransformations& removeAll() {
        removeBranchSpecific<Transformation, Operation>();
        remove<Transformation, Operation>();
        removeCleanup<Transformation, Operation>();
        removeStandaloneCleanup<Transformation, Operation>();

        return *this;
    }

    /**
     * Add branch specific transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addBranchSpecific(const LayerTransformation::Params& params) {
        const std::string typeName = getType<Operation>();
        const auto it = branchSpecificTransformations.find(typeName);
        if (it != branchSpecificTransformations.end()) {
            branchSpecificTransformations.erase(it);
        }

        branchSpecificTransformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    /**
    * Add decomposition transformation. Transformation type and operation type are required.
    * Operation type is used to find transformation by operation during precision definition.
    */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addDecomposition(const LayerTransformation::Params& params) {
        const std::string typeName = getType<Operation>();
        const auto it = decompositionTransformations.find(typeName);
        if (it != decompositionTransformations.end()) {
            decompositionTransformations.erase(it);
        }

        decompositionTransformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    /**
     * Add transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& add(const LayerTransformation::Params& params) {
        const std::string typeName = getType<Operation>();
        const auto it = transformations.find(typeName);
        if (it != transformations.end()) {
            transformations.erase(it);
        }

        transformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    /**
     * Add cleanup transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addCleanup(const LayerTransformation::Params& params) {
        const std::string typeName = getType<Operation>();
        const std::string typeId = typeid(Transformation).name();
        const auto it = cleanupTransformations.find(typeName);
        if (it == cleanupTransformations.end()) {
            cleanupTransformations.emplace(typeName,
                std::vector<std::pair<std::string, LayerTransformationPtr>>{ std::make_pair(typeId, std::make_shared<Transformation>(params)) });
        } else {
            const auto it1 = std::find_if(it->second.begin(), it->second.end(),
                [&](const std::pair<std::string, LayerTransformationPtr>& transformation) {
                    return transformation.first == typeName;
                });
            if (it1 != it->second.end()) {
                it->second.erase(it1);
            }
            it->second.emplace_back(std::make_pair(typeId, std::make_shared<Transformation>(params)));
        }
        return *this;
    }

    /**
     * Add cleanup transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addStandaloneCleanup(const LayerTransformation::Params& params) {
        const std::string typeName = getType<Operation>();
        const std::string typeId = typeid(Transformation).name();
        const auto it = std::find_if(standaloneCleanupTransformations.begin(), standaloneCleanupTransformations.end(),
            [&](const StandaloneCleanup& transformation) {
                return transformation.typeName == typeName && transformation.typeId == typeId;
            });
        if (it == standaloneCleanupTransformations.end()) {
            standaloneCleanupTransformations.emplace_back(StandaloneCleanup{ typeName, typeId, std::make_shared<Transformation>(params) });
        } else {
            *it = { typeName, typeId, std::make_shared<Transformation>(params) };
        }

        return *this;
    }

    template <class Operation>
    static std::string getType() {
        return Operation::get_type_info_static().name;
    }

    static std::string getType(const Node& operation) {
        return operation.get_type_name();
    }

    std::vector<LayerTransformationPtr> find(const std::string& transformationName) const;

    template <class Operation>
    std::vector<LayerTransformationPtr> find() const {
        const std::string transformationKey = getType<Operation>();
        return find(transformationKey);
    }

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    // Key is not a layer type, but just a name of transformation
    // Layer type (or a pattern) is defined by transformation itself as an ngraph matcher
    std::map<std::string, LayerTransformationPtr> branchSpecificTransformations;
    std::map<std::string, LayerTransformationPtr> decompositionTransformations;
    std::map<std::string, LayerTransformationPtr> transformations;
    std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>> cleanupTransformations;
    std::vector<StandaloneCleanup> standaloneCleanupTransformations;

private:
    static void setParamsManager(IParamsManager* paramsManager, std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
    static void setParamsManager(
        IParamsManager* paramsManager,
        std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& transformations) noexcept;
    static void setParamsManager(IParamsManager* paramsManager, std::vector<StandaloneCleanup>& transformations) noexcept;
    static void setLayerTransformationsManager(
        ILayerTransformationsManager* layerTransformationsManager,
        std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
    static void setLayerTransformationsManager(
        ILayerTransformationsManager* layerTransformationsManager,
        std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>>& transformations) noexcept;
    static void setLayerTransformationsManager(
        ILayerTransformationsManager* layerTransformationsManager,
        std::vector<StandaloneCleanup>& transformations) noexcept;
};

/**
 * @brief low precision transformation component.
  */
class TRANSFORMATIONS_API LowPrecisionTransformer : public IParamsManager, ILayerTransformationsManager {
public:
    static LowPrecisionTransformations getAllTransformations(const LayerTransformation::Params& params = LayerTransformation::Params());

    static bool isFunctionQuantized(const std::shared_ptr<const Function>& function);

    LowPrecisionTransformer();
    LowPrecisionTransformer(const LowPrecisionTransformations& transformations);
    void transform(std::shared_ptr<Function> network);

    // IParamsManager interface implementation
    std::vector<element::Type> getPrecisionsOnActivations(const Node& op) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(const std::shared_ptr<Node>& layer) const noexcept override;
    bool isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept override;

private:
    LowPrecisionTransformations transformations;

    void registerAllMatchers(
        std::map<std::string, LayerTransformationPtr> transformations,
        GraphRewrite& pass,
        TransformationContext& context);

    void registerAllMatchers(
        std::map<std::string, std::vector<std::pair<std::string, LayerTransformationPtr>>> transformations,
        GraphRewrite& pass,
        TransformationContext& context);
};

class TRANSFORMATIONS_API TypeRelaxedReplacer : public GraphRewrite {
public:
    TypeRelaxedReplacer();
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
