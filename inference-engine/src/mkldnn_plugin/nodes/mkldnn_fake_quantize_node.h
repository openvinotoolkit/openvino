// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <common/primitive_attr.hpp>

#include <string>
#include <memory>
#include <vector>
#include <utility>

namespace MKLDNNPlugin {

struct jit_quantize_params {
    int c;

    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision wei_prc;
    InferenceEngine::Precision dst_prc;

    InferenceEngine::Layout src_layout;

    Algorithm op_type;
};

struct jit_quantize_call_args {
    const uint8_t* from;
    const uint8_t* to;
    const float* thresholds;
    const float* output_mask;

    const float* crop_low;
    const float* crop_high;
    const float* input_scale;
    const float* input_shift;
    const float* output_scale;
    const float* output_shift;

    size_t src_step;
    size_t dst_step;
    size_t block_size;
    size_t work_amount;
};

struct jit_uni_quantize_kernel {
    void (*ker_)(const jit_quantize_call_args *);

    void operator()(const jit_quantize_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_quantize_kernel(jit_quantize_params jqp) : ker_(nullptr), jqp_(jqp) {}
    virtual ~jit_uni_quantize_kernel() {}

    virtual void create_ker() = 0;

    jit_quantize_params jqp_;
};

class MKLDNNFakeQuantizeNode : public MKLDNNNode {
public:
    MKLDNNFakeQuantizeNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    void createPrimitive() override;
    bool created() const override;
    void execute(mkldnn::stream strm) override;

    size_t getAxis() const { return axis; }

    bool isBinarization() const { return getAlgorithm() == Algorithm::FQBinarization; }

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

    void setCropLow(std::vector<float> newCropLow) { cropLow = std::move(newCropLow); isPostOpDataInitialized = false; }
    void setCropHigh(std::vector<float> newCropHigh) { cropHigh = std::move(newCropHigh); isPostOpDataInitialized = false; }
    void setInputScale(std::vector<float> newInputScale) { inputScale = std::move(newInputScale); isPostOpDataInitialized = false; }
    void setInputShift(std::vector<float> newInputShift) { inputShift = std::move(newInputShift); isPostOpDataInitialized = false; }
    void setOutputScale(std::vector<float> newOutputScale) { outputScale = std::move(newOutputScale); isPostOpDataInitialized = false;}
    void setOutputShift(std::vector<float> newOutputShift) { outputShift = std::move(newOutputShift); isPostOpDataInitialized = false; }

    bool isInputLowBroadcast() const { return isInputLowBroadcasted; }
    bool isInputHighBroadcast() const { return isInputHighBroadcasted; }
    bool isOutputLowBroadcast() const { return isOutputLowBroadcasted; }
    bool isOutputHighBroadcast() const { return isOutputHighBroadcasted; }

    InferenceEngine::Precision getInputPrecision() const { return inputPrecision; }
    InferenceEngine::Precision getOutputPrecision() const { return outputPrecision; }

    void appendPostOps(mkldnn::post_ops& ops) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    void init() override;
    std::vector<mkldnn::memory::format_tag> getDataFormats() const;
    void executeReference();
    void executeBinarization();
    void executeQuantization();

    size_t levels = 0;

    bool binarization = false;

    std::vector<float> binarizationThresholds;
    std::vector<uint32_t> binarizationOutputMask;

    std::vector<float> cropLow;
    std::vector<float> cropHigh;
    std::vector<float> inputScale;
    std::vector<float> inputShift;
    std::vector<float> outputScale;
    std::vector<float> outputShift;

    // mkldnn style post ops data representation
    bool isPostOpDataInitialized = false;
    mkldnn::impl::shifts_t<float> cropLowData;
    mkldnn::impl::shifts_t<float> cropHighData;
    mkldnn::impl::scales_t inputScaleData;
    mkldnn::impl::shifts_t<float> inputShiftData;
    mkldnn::impl::scales_t outputScaleData;
    mkldnn::impl::shifts_t<float> outputShiftData;

    bool isInputLowBroadcasted = false;
    bool isInputHighBroadcasted = false;
    bool isOutputLowBroadcasted = false;
    bool isOutputHighBroadcasted = false;

    size_t axis = 0;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    jit_quantize_params jqp = {};

    std::shared_ptr<jit_uni_quantize_kernel> quantize_kernel = nullptr;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
