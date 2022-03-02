// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <common/primitive_attr.hpp>

#include <string>
#include <memory>
#include <vector>
#include <utility>

namespace ov {
namespace intel_cpu {
namespace node {

struct jit_quantize_params {
    int c;
    bool is_planar;

    InferenceEngine::Precision src_prc;
    InferenceEngine::Precision wei_prc;
    InferenceEngine::Precision dst_prc;

    std::vector<size_t> s_str;
    std::vector<size_t> d_str;

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

    explicit jit_uni_quantize_kernel(const jit_quantize_params& jqp) : ker_(nullptr), jqp_(jqp) {}
    virtual ~jit_uni_quantize_kernel() {}

    virtual void create_ker() = 0;

    jit_quantize_params jqp_;
};

class FakeQuantize : public Node {
public:
    FakeQuantize(const std::shared_ptr<ngraph::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    bool created() const override;
    void execute(dnnl::stream strm) override;
    void executeDynamicImpl(dnnl::stream strm) override;

    size_t getAxis() const { return axis; }

    bool isBinarization() const { return getAlgorithm() == Algorithm::FQBinarization; }

    bool needPrepareParams() const override;
    void prepareParams() override;

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

    void setCropLow(std::vector<float> newCropLow) {
        cropLow = std::move(newCropLow); cropLowSize = cropLow.size(); isPostOpDataInitialized = false;
    }
    void setCropHigh(std::vector<float> newCropHigh) {
        cropHigh = std::move(newCropHigh); cropHighSize = cropHigh.size(); isPostOpDataInitialized = false;
    }
    void setInputScale(std::vector<float> newInputScale) {
        inputScale = std::move(newInputScale); inputScaleSize = inputScale.size(); isPostOpDataInitialized = false;
    }
    void setInputShift(std::vector<float> newInputShift) {
        inputShift = std::move(newInputShift); inputShiftSize = inputShift.size(); isPostOpDataInitialized = false;
    }
    void setOutputScale(std::vector<float> newOutputScale) {
        outputScale = std::move(newOutputScale); outputScaleSize = outputScale.size(); isPostOpDataInitialized = false;
    }
    void setOutputShift(std::vector<float> newOutputShift) {
        outputShift = std::move(newOutputShift); outputShiftSize = outputShift.size(); isPostOpDataInitialized = false;
    }

    bool isInputLowBroadcast() const { return isInputLowBroadcasted; }
    bool isInputHighBroadcast() const { return isInputHighBroadcasted; }
    bool isOutputLowBroadcast() const { return isOutputLowBroadcasted; }
    bool isOutputHighBroadcast() const { return isOutputHighBroadcasted; }

    InferenceEngine::Precision getInputPrecision() const { return inputPrecision; }
    InferenceEngine::Precision getOutputPrecision() const { return outputPrecision; }

    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<MemoryPtr>& postOpsMem, const size_t channelAxis = 1) override;
    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<const void*>& postOpsMem, const size_t channelAxis = 1) override;
    void appendBinPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<MemoryPtr>& binaryPostOpsMem) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

    enum BroadcastingPolicy {
        PerChannel, // all FQ operations are per channel
        PerTensor,  // all FQ operations are per tensor
        Mixed,      // some per channel, some per tensor
    };

    BroadcastingPolicy getBroadcastingPolicy() const { return broadcastingPolicy; }

    MemoryPtr cropLowMemory;
    MemoryPtr cropHighMemory;
    MemoryPtr inputScaleMemory;
    MemoryPtr inputShiftMemory;
    MemoryPtr outputScaleMemory;
    MemoryPtr outputShiftMemory;

private:
    struct FakeQuantizeExecutor {
        virtual void exec(const FakeQuantize& node) = 0;
        virtual ~FakeQuantizeExecutor() = default;
    };
    using executorPtr = std::shared_ptr<FakeQuantizeExecutor>;
    executorPtr execPtr = nullptr;

    struct FakeQuantizeJitExecutor : public FakeQuantizeExecutor {
        FakeQuantizeJitExecutor(const jit_quantize_params &_jqp);
        void exec(const FakeQuantize& node) override;
        std::unique_ptr<jit_uni_quantize_kernel> pKernel;
    };

    void init() override;
    std::vector<LayoutType> getDataFormats() const;
    void initializePostOpData(const VectorDims &postOpDims, const size_t bufferAlignment);
    void initializePostOpDataLegacy(const VectorDims &dims, const size_t bufferAlignment);
    void executeReference();
    void executeBinarization(const std::unique_ptr<jit_uni_quantize_kernel> &pKernel) const;
    void executeQuantization(const std::unique_ptr<jit_uni_quantize_kernel> &pKernel) const;

    void appendMemory(const size_t dataSize, const void *data, MemoryPtr &memPtr, std::vector<MemoryPtr>& postOpsMem);
    void appendMemory(const size_t dataSize, const void *data, MemoryPtr &memPtr, std::vector<const void*>& postOpsMem);
    template <typename T>
    void appendPostOpsImpl(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<T>& postOpsMem);

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

    std::vector<float> quantizationData;
    size_t quantizationDataSize = 0lu;
    MemoryPtr quantizationMemory;

    size_t cropLowSize;
    size_t cropHighSize;
    size_t inputScaleSize;
    size_t inputShiftSize;
    size_t outputScaleSize;
    size_t outputShiftSize;

    // onednn style post ops data representation
    bool isPostOpDataInitialized = false;
    dnnl::impl::shifts_t<float> cropLowData;
    dnnl::impl::shifts_t<float> cropHighData;
    dnnl::impl::scales_t inputScaleData;
    dnnl::impl::shifts_t<float> inputShiftData;
    dnnl::impl::scales_t outputScaleData;
    dnnl::impl::shifts_t<float> outputShiftData;

    bool isInputLowBroadcasted = false;
    bool isInputHighBroadcasted = false;
    bool isOutputLowBroadcasted = false;
    bool isOutputHighBroadcasted = false;

    size_t currentAxisSize = 0;
    size_t axis = 0;

    InferenceEngine::Precision inputPrecision = InferenceEngine::Precision::FP32;
    InferenceEngine::Precision outputPrecision = InferenceEngine::Precision::FP32;

    std::string errorPrefix;

    BroadcastingPolicy broadcastingPolicy;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
