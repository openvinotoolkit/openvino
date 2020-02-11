// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file
 */

#pragma once

#include <builders/ie_layer_decorator.hpp>
#include <ie_network.hpp>
#include <string>
#include <vector>

namespace InferenceEngine {
namespace Builder {

/**
 * @deprecated Use ngraph API instead.
 * @brief The class represents a builder for Resample layer
 */
IE_SUPPRESS_DEPRECATED_START
class INFERENCE_ENGINE_NN_BUILDER_API_CLASS(ResampleLayer): public LayerDecorator {
public:
    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit ResampleLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer pointer to generic builder
     */
    explicit ResampleLayer(const Layer::Ptr& layer);
    /**
     * @brief The constructor creates a builder from generic builder
     * @param layer const pointer to generic builder
     */
    explicit ResampleLayer(const Layer::CPtr& layer);
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    ResampleLayer& setName(const std::string& name);

    /**
     * @brief Returns input port
     * @return Input port
     */
    const Port& getInputPort() const;
    /**
     * @brief Sets input port
     * @param ports Input port
     * @return reference to layer builder
     */
    ResampleLayer& setInputPort(const Port& ports);
    /**
     * @brief Returns output port
     * @return Output port
     */
    const Port& getOutputPort() const;
    /**
     * @brief Sets output port
     * @param port Output port
     * @return reference to layer builder
     */
    ResampleLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns resample type
     * @return Type
     */
    const std::string& getResampleType() const;
    /**
     * @brief Sets resample type
     * @param type Type
     * @return reference to layer builder
     */
    ResampleLayer& setResampleType(const std::string& type);
    /**
     * @brief Returns flag that denotes whether to perform anti-aliasing
     * @return true if anti-aliasing is performed
     */
    bool getAntialias() const;
    /**
     * @brief Sets flag that denotes whether to perform anti-aliasing
     * @param flag antialias
     * @return reference to layer builder
     */
    ResampleLayer& setAntialias(bool antialias);
    /**
     * @brief Returns resample factor
     * @return Factor
     */
    float getFactor() const;
    /**
     * @brief Sets resample factor
     * @param factor Factor
     * @return reference to layer builder
     */
    ResampleLayer& setFactor(float factor);
    /**
     * @brief Returns width
     * @return Width
     */
    size_t getWidth() const;
    /**
     * @brief Sets width
     * @param width Width
     * @return reference to layer builder
     */
    ResampleLayer& setWidth(size_t width);
    /**
     * @brief Returns height
     * @return Height
     */
    size_t getHeight() const;
    /**
     * @brief Sets height
     * @param height Height
     * @return reference to layer builder
     */
    ResampleLayer& setHeight(size_t height);
};
IE_SUPPRESS_DEPRECATED_END

}  // namespace Builder
}  // namespace InferenceEngine
