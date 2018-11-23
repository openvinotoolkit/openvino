// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <list>
#include <map>
#include <set>
#include <memory>

#include <ie_layers.h>
#include "details/caseless.hpp"
#include "shape_infer/built-in/ie_built_in_holder.hpp"
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
    virtual ReshapeLauncher::Ptr
    createNotInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions);

    /**
     * @brief Creates reshape launcher for the given input layer. Supported types: Input, Const, Memory (as input)
     * @param layer - const pointer to the CNNLayer for which shape infer is needed
     * @param extensions - all registered extensions
     * @return - shared_ptr to the corresponding launcher.
     */
    virtual ReshapeLauncher::Ptr
    createInputLauncher(const CNNLayer* layer, const std::vector<IShapeInferExtensionPtr>& extensions);

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
    explicit Reshaper(ICNNNetwork& network,
                      const LauncherCreator::Ptr& creator = std::make_shared<LauncherCreator>());

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
    void run(const std::map<std::string, SizeVector>& inputShapes);

    using Ptr = std::shared_ptr<Reshaper>;
private:
    ReshapeLauncher::Ptr getLauncherByLayerName(const std::string& layerName) const;

    static InferenceEngine::details::caseless_set<std::string> getTypeNamesFromExtension(const IShapeInferExtensionPtr& extension);

private:
    std::vector<IShapeInferExtensionPtr> _extensions;
    std::set<ReshapeLauncher::Ptr> _launchers;
    std::vector<CNNLayerPtr> _allSortedLayers{};
    std::set<CNNLayerPtr> _inputLayers{};
    InferenceEngine::details::caseless_set<std::string> _allTypes;
};

}  // namespace ShapeInfer
}  // namespace InferenceEngine
