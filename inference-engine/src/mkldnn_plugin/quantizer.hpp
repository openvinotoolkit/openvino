// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <map>
#include <vector>

#include <ie_icnn_network.hpp>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {
namespace details {

/**
 * @brief Quantization layer details and basic operations on them.
  */
class QuantizationDetails {
public:
    QuantizationDetails();
    QuantizationDetails(const QuantizationDetails& quantizationDetails);
    QuantizationDetails(
        const size_t levels,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues);

    static QuantizationDetails getDetails(const CNNLayerPtr quantize);
    bool hasNegativeOutput() const;
    float maxOutput(const size_t channel) const;
    float maxInput(const size_t channel) const;
    size_t inputChannels() const;
    size_t outputChannels() const;

    const size_t levels;
    const std::vector<float> inputLowValues;
    const std::vector<float> inputHighValues;
    const std::vector<float> outputLowValues;
    const std::vector<float> outputHighValues;

private:
    static void validate(const CNNLayerPtr& constantLayer);
    static std::vector<float> getBlobValue(const CNNLayerPtr& constantLayer);
};

/**
 * @brief Quantization class to quantize ICNNNetwork.
  */
class Quantizer {
public:
    enum QuantizationPrecision {
        NONE = 0,
        FP32 = 1,
        INT8 = 2
    };

    /**
     * Check if network can be quantized or not.
     * @return true if network can be quantized or false if not
     */
    static bool isNetworkSupported(const ICNNNetwork& network);

    /**
     * Quantize network.
     * @param dotFilePath dot file path for exploration proposals. Empty string is default value.
     * @param irFilePath model file path after weights quantization for exploration proposals. Empty string is default value.
     */
    void quantize(
        ICNNNetwork& network,
        const QuantizationPrecision precision = QuantizationPrecision::INT8,
        const bool transformQuantizeOnDataPath = true,
        const std::string& dotFilePath = "",
        const std::string& irFilePath = "");

    /**
     * Handle Quantize layers on weights path.
     */
    void transformQuantizeOnWeightsPath(
        ICNNNetwork& network,
        const std::string& irFilePath = "");

private:
    void transformQuantizeOnWeightsPath(ICNNNetwork& network, CNNLayerPtr layer);

    /**
     * Calculate layer properties to infer laer in low precision.
     * @param network which contains layer
     * @param quantize Quantize layer
     * @param layer layer for quantization
     * @param mode quantization mode
     * @param quantizationDetails map with quantization details
     * @param transformQuantizeOnDataPath boolean flag to transform Quantize layers on data path or not
     */
    static void quantizeConvolutionOrFullyConnected(
        ICNNNetwork& network,
        CNNLayerPtr& quantize,
        CNNLayerPtr& layer,
        QuantizationPrecision mode,
        std::map<std::string, const QuantizationDetails>& quantizationDetails,
        const bool transformQuantizeOnDataPath = true);

    /**
     * Extract and quantize weights.
     */
    static Blob::Ptr quantizeWeights(const CNNLayerPtr quantize);

    static int getMaxSignValue(const size_t quantizationLevels);
    static int getMaxUnsignValue(const size_t quantizationLevels);

    static void definesExecutionPrecision(
        ICNNNetwork& net,
        const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap);

    static void propagateScaleFactors(ICNNNetwork& net);

    static Blob::Ptr calculateScale(const QuantizationDetails& quantizationDetails, const size_t channels);

    static void ScaleDataToInt(
        const float* srcData,
        size_t srcSize,
        Blob::Ptr int8blob,
        const std::vector<float>& scales);
};

typedef std::shared_ptr<Quantizer> QuantizerPtr;

}  // namespace details
}  // namespace InferenceEngine
