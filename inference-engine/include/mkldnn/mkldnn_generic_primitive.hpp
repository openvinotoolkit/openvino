// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for MKL-DNN Generic Primitive API
 * @file mkldnn_generic_primitive.hpp
 */
#pragma once

#include "mkldnn_extension_types.hpp"
#include "details/ie_irelease.hpp"
#include <vector>

namespace InferenceEngine {
namespace MKLDNNPlugin {

/**
 * @deprecated use new extensibility API
 * @brief The MKLDNNGenericFormats stores weights, biases, inputs and outputs of the primitive
 */
class MKLDNNGenericFormats {
public:
    /**
     * @brief A default constructor
     * @param ins - vector of inputs
     * @param outs - vector of outputs
     * @param weights - weights, format_undef by default
     * @param biases  - biases, format_undef by default
     */
    MKLDNNGenericFormats(const std::vector<MemoryFormat> &ins, const std::vector<MemoryFormat> &outs,
                         const MemoryFormat weights = MemoryFormat::format_undef,
                         const MemoryFormat biases = MemoryFormat::format_undef) : inputs(ins), outputs(outs) {
        this->weights = weights;
        this->biases = biases;
    }

    /**
     * @brief Get input formats
     * @return vector of input formats
     */
    const std::vector<MemoryFormat>& GetInputs() const noexcept {
        return inputs;
    }

    /**
     * @brief Get output formats
     * @return vector of output formats
     */
    const std::vector<MemoryFormat>& GetOutputs() const noexcept {
        return outputs;
    }

    /**
     * @brief Get weights format
     * @return weights format
     */
    const MemoryFormat& GetWeights() const noexcept {
        return weights;
    }

    /**
     * @brief Get biases format
     * @return biases format
     */
    const MemoryFormat& GetBiases() const noexcept {
        return biases;
    }

private:
    std::vector<MemoryFormat> inputs;
    std::vector<MemoryFormat> outputs;
    MemoryFormat weights;
    MemoryFormat biases;
};

/**
 * @deprecated use new extensibility API
 * @brief The IMKLDNNGenericPrimitive is the main Generic Primitive interface
 */
class IMKLDNNGenericPrimitive : public InferenceEngine::details::IRelease {
public:
    void Release() noexcept override {
        delete this;
    }

    /**
     * @brief Sets inputs nd outputs
     * @param inputs - vector of input primitives
     * @param outputs - vector of output primitives
     */
    void SetMemory(const std::vector<MKLDNNPrimitiveMemory>& inputs,
                           const std::vector<MKLDNNPrimitiveMemory>& outputs) noexcept {
        this->inputs = inputs;
        this->outputs = outputs;
    }

    /**
     * @brief Gets supported formats
     * @return vector of supported formats
     */
    virtual std::vector<MKLDNNGenericFormats> GetSupportedFormats() noexcept = 0;

    /**
     * @brief Entry point of actual execution of primitive.
     * Error reporting mechanism missed, static check should be done in constructor
     */
    virtual void Execute() noexcept = 0;

protected:
    /**
     * @brief Vector of input primitives
     */
    std::vector<MKLDNNPrimitiveMemory> inputs;
    /**
     * @brief Vector of output primitives
     */
    std::vector<MKLDNNPrimitiveMemory> outputs;
};

}  // namespace MKLDNNPlugin
}  // namespace InferenceEngine
