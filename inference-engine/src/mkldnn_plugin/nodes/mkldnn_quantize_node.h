// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNQuantizeNode : public MKLDNNNode {
public:
    MKLDNNQuantizeNode(InferenceEngine::CNNLayerPtr layer, const mkldnn::engine& eng, int socket);
    ~MKLDNNQuantizeNode() override = default;

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    const float* getBinarizationTresholdsPtr() {
        if (!initialized)
            initValues();
        return &binarizationThresholds[0];
    }

    size_t getBinarizationTresholdsSize() {
        if (!initialized)
            initValues();
        return binarizationThresholds.size();
    }

    const float* getBinarizationOutputMaskPtr() {
        if (!initialized)
            initValues();
        return reinterpret_cast<float*>(&binarizationOutputMask[0]);
    }

    size_t getBinarizationOutputMaskSize() {
        if (!initialized)
            initValues();
        return binarizationOutputMask.size();
    }

    bool isBinarization() {
        if (!initialized)
            initValues();
        return quantizeAlgorithm == mkldnn::algorithm::binarization_depthwise;
    }

    size_t getAxis() {
        if (!initialized)
            initValues();
        return axis;
    }

    const float* getCropLowPtr() {
        if (!initialized)
            initValues();
        return &cropLow[0];
    }

    const float* getCropHighPtr() {
        if (!initialized)
            initValues();
        return &cropHigh[0];
    }

    const float* getInputScalePtr() {
        if (!initialized)
            initValues();
        return &inputScale[0];
    }

    const float* getInputShiftPtr() {
        if (!initialized)
            initValues();
        return &inputShift[0];
    }

    const float* getOutputScalePtr() {
        if (!initialized)
            initValues();
        return &outputScale[0];
    }

    const float* getOutputShiftPtr() {
        if (!initialized)
            initValues();
        return &outputShift[0];
    }

    InferenceEngine::Precision getInputPrecision() {
        if (!initialized)
            initValues();
        return inputPrecision;
    }

    InferenceEngine::Precision getOutputPrecision() {
        if (!initialized)
            initValues();
        return outputPrecision;
    }

    mkldnn::algorithm getAlgorithm() {
        return quantizeAlgorithm;
    }

private:
    void initValues();
    std::vector<mkldnn::memory::format> getDataFormats();

    bool initialized = false;
    int levels = -1;

    std::vector<float> binarizationThresholds;
    std::vector<uint32_t> binarizationOutputMask;

    std::vector<float> cropLow;
    std::vector<float> cropHigh;
    std::vector<float> inputScale;
    std::vector<float> inputShift;
    std::vector<float> outputScale;
    std::vector<float> outputShift;

    size_t inputLowAxis = 0;
    size_t inputHighAxis = 0;
    size_t outputLowAxis = 0;
    size_t outputHighAxis = 0;

    size_t axis = 0;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    mkldnn::algorithm quantizeAlgorithm = mkldnn::algorithm::algorithm_undef;
};

}  // namespace MKLDNNPlugin
