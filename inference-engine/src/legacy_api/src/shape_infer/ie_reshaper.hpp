// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "caseless.hpp"
#include "ie_icnn_network.hpp"

#include <legacy/ie_ishape_infer_extension.hpp>
#include <legacy/ie_layers.h>
#include "ie_reshape_launcher.hpp"

namespace InferenceEngine {
namespace ShapeInfer {

class INFERENCE_ENGINE_API_CLASS(LauncherCreator) {
public:
    using Ptr = std::shared_ptr<LauncherCreator>;

    /**
     * @brief Creates reshape launcher for the given intermediate layer with first registered implementation.
     * Built-in implementations are first, then - custom ones.
     * Throws exception if it fails to find implementation for the given layer.
     * @param layer - const pointer to the CNNLayer for which shape infer is needed
     * @param extensions - all registered extensions
     * @return - shared_ptr to the corresponding launcher.
     */
    virtual ReshapeLauncher::Ptr createNotInputLauncher(const CNNLayer* layer,
                                                        const std::vector<IShapeInferExtensionPtr>& extensions);

    /**
     * @brief Creates reshape launcher for the given input layer. Supported types: Input, Const, Memory (as input)
     * @param layer - const pointer to the CNNLayer for which shape infer is needed
     * @param extensions - all registered extensions
     * @return - shared_ptr to the corresponding launcher.
     */
    virtual ReshapeLauncher::Ptr createInputLauncher(const CNNLayer* layer,
                                                     const std::vector<IShapeInferExtensionPtr>& extensions);

    virtual ~LauncherCreator() = default;
};

/**
 * @class Reshaper
 * @brief Helper class to infer shapes for the given ICNNNetwork.
 * It delegates shape inference to the corresponding ReshapeLauncher.
 */
class INFERENCE_ENGINE_API_CLASS(Reshaper) {
public:
    /**
     * @brief Constructor
     * @param network - const reference to the ICNNNetwork for performing shape inference
     */
    explicit Reshaper(ICNNNetwork& network, const LauncherCreator::Ptr& creator = std::make_shared<LauncherCreator>());

    explicit Reshaper(std::vector<DataPtr> inputs,
                      const LauncherCreator::Ptr& launcherCreator = std::make_shared<LauncherCreator>());

    virtual ~Reshaper() = default;

    /**
     * @brief Adds shape infer extension to provide implementations of shape infer functions
     * @param extension - pointer to the shape infer extension
     */
    void AddExtension(const IShapeInferExtensionPtr& extension);

    /**
     * @brief Launches shape inference for the given ICNNNetworkAdds and input shapes.
     * Throws if shape infer failed without corruption of original shapes
     * @param inputShapes - Map of input names (data) to their input shapes.
     */
    StatusCode run(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp = nullptr);

    /**
     * @brief Perform shape inference for the given input shapes but not apply it.
     * In case of success call apply() method.
     * @param inputShapes - Map of input names (data) to their input shapes.
     * @throws exception if shape infer failed without corruption of original shapes
     */
    StatusCode runNoApply(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp = nullptr);

    /**
     * @brief Apply shapes pre calculated by runNoApply() method.
     */
    StatusCode apply(ResponseDesc* resp = nullptr);

    /**
     * @brief Return newly calculated shape for provided data.
     */
    SizeVector getResultShapeFor(DataPtr& data, ResponseDesc* resp = nullptr);

private:
    ReshapeLauncher::Ptr getLauncherByLayerName(const std::string& layerName) const;

    InferenceEngine::details::caseless_set<std::string> getTypeNamesFromExtension(
        const IShapeInferExtensionPtr& extension);

    std::vector<IShapeInferExtensionPtr> _extensions;
    std::set<ReshapeLauncher::Ptr> _launchers;
    std::vector<CNNLayerPtr> _allSortedLayers {};
    std::set<CNNLayerPtr> _inputLayers {};
    InferenceEngine::details::caseless_set<std::string> _allTypes;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
