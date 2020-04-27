// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include <utility>

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

    size_t getAxis() const { return axis; }

    bool isBinarization() const { return quantizeAlgorithm == mkldnn::algorithm::binarization_depthwise; }
    mkldnn::algorithm getAlgorithm() const { return quantizeAlgorithm; }

    const float* getBinarizationTresholdsPtr() const { return &binarizationThresholds[0]; }
    const float* getBinarizationOutputMaskPtr() const { return reinterpret_cast<const float*>(&binarizationOutputMask[0]); }
    size_t getBinarizationTresholdsSize() const { return binarizationThresholds.size(); }
    size_t getBinarizationOutputMaskSize() const { return binarizationOutputMask.size(); }

    const std::vector<float>& getCropLow() const { return cropLow; }
    const std::vector<float>& getCropHigh() const { return cropHigh; }
    const std::vector<float>& getInputScale() const { return inputScale; }
    const std::vector<float>& getInputShift() const { return inputShift; }
    const std::vector<float>& getOutputScale() const { return outputScale; }
    const std::vector<float>& getOutputShift() const { return outputShift; }

    void setCropLow(std::vector<float> newCropLow) { cropLow = std::move(newCropLow); }
    void setCropHigh(std::vector<float> newCropHigh) { cropHigh = std::move(newCropHigh); }
    void setInputScale(std::vector<float> newInputScale) { inputScale = std::move(newInputScale); }
    void setInputShift(std::vector<float> newInputShift) { inputShift = std::move(newInputShift); }
    void setOutputScale(std::vector<float> newOutputScale) { outputScale = std::move(newOutputScale); }
    void setOutputShift(std::vector<float> newOutputShift) { outputShift = std::move(newOutputShift); }

    const bool isInputLowBroadcast() const { return isInputLowBroadcasted; }
    const bool isInputHighBroadcast() const { return isInputHighBroadcasted; }
    const bool isOutputLowBroadcast() const { return isOutputLowBroadcasted; }
    const bool isOutputHighBroadcast() const { return isOutputHighBroadcasted; }

    InferenceEngine::Precision getInputPrecision() const { return inputPrecision; }
    InferenceEngine::Precision getOutputPrecision() const { return outputPrecision; }

    void appendPostOps(mkldnn::post_ops& ops) override;

private:
    void init() override;
    std::vector<mkldnn::memory::format> getDataFormats() const;

    int levels = -1;

    std::vector<float> binarizationThresholds;
    std::vector<uint32_t> binarizationOutputMask;

    std::vector<float> cropLow;
    std::vector<float> cropHigh;
    std::vector<float> inputScale;
    std::vector<float> inputShift;
    std::vector<float> outputScale;
    std::vector<float> outputShift;

    bool isInputLowBroadcasted = false;
    bool isInputHighBroadcasted = false;
    bool isOutputLowBroadcasted = false;
    bool isOutputHighBroadcasted = false;

    size_t axis = 0;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    mkldnn::algorithm quantizeAlgorithm = mkldnn::algorithm::algorithm_undef;
};

}  // namespace MKLDNNPlugin
