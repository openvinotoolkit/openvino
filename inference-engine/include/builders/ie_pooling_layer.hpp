// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <builders/ie_layer_fragment.hpp>
#include <ie_inetwork.hpp>
#include <vector>
#include <string>

namespace InferenceEngine {
namespace Builder {

/**
 * @brief The class represents a builder for Pooling layer
 */
class INFERENCE_ENGINE_API_CLASS(PoolingLayer): public LayerFragment {
public:
    /**
     * @brief The enum defines available pooling types
     */
    enum PoolingType {
        MAX = 1,
        AVG = 2
    };

    /**
     * @brief The enum defines available rounding types
     */
    enum RoundingType {
        CEIL = 1,
        FLOOR = 2
    };

    /**
     * @brief The constructor creates a builder with the name
     * @param name Layer name
     */
    explicit PoolingLayer(const std::string& name = "");
    /**
     * @brief The constructor creates a builder from generic builder
     * @param genLayer generic builder
     */
    explicit PoolingLayer(Layer& genLayer);
    /**
     * @brief Operator creates generic layer builder
     * @return Generic layer builder
     */
    operator Layer() const override;
    /**
     * @brief Sets the name for the layer
     * @param name Layer name
     * @return reference to layer builder
     */
    PoolingLayer& setName(const std::string& name);

    /**
     * @brief Returns input port
     * @return Input port
     */
    const Port& getInputPort() const;
    /**
     * @brief Sets input port
     * @param port Input port
     * @return reference to layer builder
     */
    PoolingLayer& setInputPort(const Port& port);
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
    PoolingLayer& setOutputPort(const Port& port);
    /**
     * @brief Returns kernel size
     * @return Kernel size
     */
    const std::vector<size_t> getKernel() const;
    /**
     * @brief Sets kernel size
     * @param kernel Kernel size
     * @return reference to layer builder
     */
    PoolingLayer& setKernel(const std::vector<size_t>& kernel);
    /**
     * @brief Returns vector of strides
     * @return vector of strides
     */
    const std::vector<size_t> getStrides() const;
    /**
     * @brief Sets strides
     * @param strides vector of strides
     * @return reference to layer builder
     */
    PoolingLayer& setStrides(const std::vector<size_t>& strides);
    /**
     * @brief Returns begin paddings
     * @return vector of paddings
     */
    const std::vector<size_t> getPaddingsBegin() const;
    /**
     * @brief Sets begin paddings
     * @param paddings Vector of paddings
     * @return reference to layer builder
     */
    PoolingLayer& setPaddingsBegin(const std::vector<size_t>& paddings);
    /**
     * @brief Return end paddings
     * @return Vector of paddings
     */
    const std::vector<size_t> getPaddingsEnd() const;
    /**
     * @brief Sets end paddings
     * @param paddings Vector of paddings
     * @return reference to layer builder
     */
    PoolingLayer& setPaddingsEnd(const std::vector<size_t>& paddings);
    /**
     * @brief Returns pooling type
     * @return Pooling type
     */
    PoolingType getPoolingType() const;
    /**
     * @brief Sets pooling type
     * @param type Pooling type
     * @return reference to layer builder
     */
    PoolingLayer& setPoolingType(PoolingType type);
    /**
     * @brief Returns rounding type
     * @return Rounding type
     */
    RoundingType getRoundingType() const;
    /**
     * @brief Sets rounding types
     * @param type Rounding type
     * @return reference to layer builder
     */
    PoolingLayer& setRoundingType(RoundingType type);
    /**
     * @brief Returns a type of pooling strategy
     * @return true if zero-values in the padding are not used
     */
    bool getExcludePad() const;
    /**
     * @brief Sets a type of pooling strategy
     * @param exclude zero-values in the padding are not used if true
     * @return reference to layer builder
     */
    PoolingLayer& setExcludePad(bool exclude);

    /**
     * @brief Validates layer before creation
     * @param layer generic layer builder
     */
    static void validate(const Layer& layer);

private:
    PoolingType type;
    RoundingType roundingType;
};

}  // namespace Builder
}  // namespace InferenceEngine
