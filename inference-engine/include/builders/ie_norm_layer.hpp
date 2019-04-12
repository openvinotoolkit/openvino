// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Norm layer
 */
class INFERENCE_ENGINE_API_CLASS(NormLayer): public LayerDecorator {
public:
    /**
     * @brief The enum defines all Norm types
     */
    enum NormType {
        WITHIN_CHANNEL = 0,
        ACROSS_CHANNELS = 1
    };
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit NormLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit NormLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer constant pointer to generic builder
     */
    explicit NormLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    NormLayer& setName(const std::string& name);

    /**
     * @brief Returns port with shapes for the layer
     * @return Port with shapes
     */
    const Port& getPort() const;
    /**
     * @brief Sets port shapes for the layer
     * @param port Port with shapes
     * @return reference to layer builder
     */
    NormLayer& setPort(const Port& port);
    /**
     * @brief Returns side length of the region
     * @return Size
     */
    size_t getSize() const;
    /**
     * @brief Sets side length of the region
     * @param size Size
     * @return reference to layer builder
     */
    NormLayer& setSize(size_t size);
    /**
     * @brief Returns scaling parameter for the normalizing sum
     * @return Scaling parameter
     */
    float getAlpha() const;
    /**
     * @brief Sets scaling parameter for the normalizing sum
     * @param alpha Scaling parameter
     * @return reference to layer builder
     */
    NormLayer& setAlpha(float alpha);
    /**
     * @brief Returns exponent for the normalizing sum
     * @return Exponent
     */
    float getBeta() const;
    /**
     * @brief Sets exponent for the normalizing sum
     * @param beta Exponent
     * @return reference to layer builder
     */
    NormLayer& setBeta(float beta);
    /**
     * @brief Returns region type
     * @return true if normalizing sum is performed over adjacent channels
     */
    bool getAcrossMaps() const;
    /**
     * @brief Sets region type
     * @param acrossMap true if normalizing sum is performed over adjacent channels
     * @return reference to layer builder
     */
    NormLayer& setAcrossMaps(bool acrossMap);
    /**
     * @brief Returns region type
     * @return Norm type
     */
    NormType getRegion() const;
    /**
     * @brief Sets region type
     * @param type region type
     * @return reference to layer builder
     */
    NormLayer& setRegion(NormType type);
};

}  // namespace Builder
}  // namespace InferenceEngine
