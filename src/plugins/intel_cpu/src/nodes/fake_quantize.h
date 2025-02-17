// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "common/primitive_attr.hpp"
#include "dnnl_postops_composer_legacy.h"
#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

enum class FQ_add_input_type { CROP_LOW, CROP_HIGH, INPUT_SCALE, INPUT_SHIFT, OUTPUT_SCALE, OUTPUT_SHIFT, INPUTS_SIZE };

struct jit_quantize_params {
    bool is_planar;

    ov::element::Type src_prc;
    ov::element::Type wei_prc;
    ov::element::Type dst_prc;

    Algorithm op_type;

    int c;                                                                         // need only for binarization
    std::bitset<static_cast<size_t>(FQ_add_input_type::INPUTS_SIZE)> broadcasted;  // need only for quantization
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
    void (*ker_)(const jit_quantize_call_args*);

    void operator()(const jit_quantize_call_args* args) {
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
    FakeQuantize(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void initSupportedPrimitiveDescriptors() override;
    void getSupportedDescriptors() override;
    bool created() const override;
    void execute(const dnnl::stream& strm) override;
    void executeDynamicImpl(const dnnl::stream& strm) override;

    size_t getAxis() const {
        return axis;
    }

    bool isBinarization() const {
        return getAlgorithm() == Algorithm::FQBinarization;
    }

    bool needPrepareParams() const override;
    void prepareParams() override;
    void createPrimitive() override;

    const float* getBinarizationTresholdsPtr() const {
        return &binarizationThresholds[0];
    }
    const float* getBinarizationOutputMaskPtr() const {
        return reinterpret_cast<const float*>(&binarizationOutputMask[0]);
    }
    size_t getBinarizationTresholdsSize() const {
        return binarizationThresholds.size();
    }
    size_t getBinarizationOutputMaskSize() const {
        return binarizationOutputMask.size();
    }

    const std::vector<float>& getCropLow() const {
        return cropLow;
    }
    const std::vector<float>& getCropHigh() const {
        return cropHigh;
    }
    const std::vector<float>& getInputScale() const {
        return inputScale;
    }
    const std::vector<float>& getInputShift() const {
        return inputShift;
    }
    const std::vector<float>& getOutputScale() const {
        return outputScale;
    }
    const std::vector<float>& getOutputShift() const {
        return outputShift;
    }
    const size_t getLevels() const {
        return levels;
    }

    void setCropLow(std::vector<float> newCropLow) {
        cropLow = std::move(newCropLow);
        cropLowSize = cropLow.size();
        ++parameterVersion;
    }
    void setCropHigh(std::vector<float> newCropHigh) {
        cropHigh = std::move(newCropHigh);
        cropHighSize = cropHigh.size();
        ++parameterVersion;
    }
    void setInputScale(std::vector<float> newInputScale) {
        inputScale = std::move(newInputScale);
        inputScaleSize = inputScale.size();
        ++parameterVersion;
    }
    void setInputShift(std::vector<float> newInputShift) {
        inputShift = std::move(newInputShift);
        inputShiftSize = inputShift.size();
        ++parameterVersion;
    }
    void setOutputScale(std::vector<float> newOutputScale) {
        outputScale = std::move(newOutputScale);
        outputScaleSize = outputScale.size();
        ++parameterVersion;
    }
    void setOutputShift(std::vector<float> newOutputShift) {
        outputShift = std::move(newOutputShift);
        outputShiftSize = outputShift.size();
        ++parameterVersion;
    }

    const std::vector<float>& getFQScales() const {
        return fqScales;
    }

    bool isInputLowBroadcast() const {
        return isInputLowBroadcasted;
    }
    bool isInputHighBroadcast() const {
        return isInputHighBroadcasted;
    }
    bool isOutputLowBroadcast() const {
        return isOutputLowBroadcasted;
    }
    bool isOutputHighBroadcast() const {
        return isOutputHighBroadcasted;
    }

    ov::element::Type getInputPrecision() const {
        return inputPrecision;
    }
    ov::element::Type getOutputPrecision() const {
        return outputPrecision;
    }

    void appendPostOps(dnnl::post_ops& ops,
                       const VectorDims& postOpDims,
                       std::unordered_map<int, MemoryPtr>& postOpsMem,
                       const int channelAxis) override;
    void appendPostOps(dnnl::post_ops& ops,
                       const VectorDims& postOpDims,
                       std::vector<const void*>& postOpsMem,
                       const int channelAxis) override;
    bool appendAttrPostOps(DnnlPostOpsComposerLegacy& dnnlpoc,
                           bool isLastPostOp,
                           dnnl::memory::data_type outDataType,
                           bool allowBinary = true,
                           bool do_rounding = true);

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    enum BroadcastingPolicy {
        PerChannel,  // all FQ operations are per channel
        PerTensor,   // all FQ operations are per tensor
        Mixed,       // some per channel, some per tensor
    };

    BroadcastingPolicy getBroadcastingPolicy() const {
        return broadcastingPolicy;
    }

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
        FakeQuantizeJitExecutor(const jit_quantize_params& _jqp);
        void exec(const FakeQuantize& node) override;
        std::unique_ptr<jit_uni_quantize_kernel> pKernel;
    };
    void init() override;
    std::vector<LayoutType> getDataFormats() const;
    void initializePostOpData(const VectorDims& postOpDims, const size_t bufferAlignment, bool doRounding);
    void initializePostOpDataLegacy(const VectorDims& dims, const size_t bufferAlignment);
    void executeReference();
    void executeBinarization(const std::unique_ptr<jit_uni_quantize_kernel>& pKernel) const;
    void executeQuantization(const std::unique_ptr<jit_uni_quantize_kernel>& pKernel) const;

    void appendMemory(const size_t dataSize, const void* data, MemoryPtr& memPtr, std::vector<MemoryPtr>& postOpsMem);
    void appendMemory(const size_t dataSize, const void* data, MemoryPtr& memPtr, std::vector<const void*>& postOpsMem);
    template <typename T>
    void appendPostOpsImpl(dnnl::post_ops& ops, const VectorDims& postOpDims, std::vector<T>& postOpsMem);

    size_t levels = 0;

    bool binarization = false;

    std::vector<float> binarizationThresholds;
    std::vector<uint32_t> binarizationOutputMask;

    // inference formula strictly following definition:
    //   x1 = crop(x, cropLow, cropHigh)
    //   x2 = x1 * inputScale + inputShift
    //   x3 = round(x2)
    //    y = x3 * outputScale + outputShift
    std::vector<float> cropLow;
    std::vector<float> cropHigh;
    std::vector<float> inputScale;
    std::vector<float> inputShift;
    std::vector<float> outputScale;
    std::vector<float> outputShift;

    // equivalently, we can push crop operation through
    // input linear mapping and rounding stage, according to
    // definition of FQ, this should turn the per-OC crop into
    // a per-tensor crop2 (maybe not always)
    struct OptimizedFormula {
        std::vector<float> isc;
        std::vector<float> ish;
        std::vector<float> osc;
        std::vector<float> osh;
        std::vector<float> clo;
        std::vector<float> chi;

        void shrinkLength() {
            auto _do_shrink = [](std::vector<float>& v) {
                if (v.size() <= 1)
                    return;
                auto ref = v[0];
                if (std::all_of(v.cbegin(), v.cend(), [&](float val) {
                        return val == ref;
                    })) {
                    v.resize(1);
                }
            };
            _do_shrink(isc);
            _do_shrink(ish);
            _do_shrink(clo);
            _do_shrink(chi);
            _do_shrink(osc);
            _do_shrink(osh);
        }
    } optimizedFormula;

    void updateOptimizedFormula(bool do_rounding);

    std::vector<float> quantizationData;
    size_t quantizationDataSize = 0lu;
    MemoryPtr quantizationMemory;

    size_t cropLowSize;
    size_t cropHighSize;
    size_t inputScaleSize;
    size_t inputShiftSize;
    size_t outputScaleSize;
    size_t outputShiftSize;

    std::bitset<static_cast<size_t>(FQ_add_input_type::INPUTS_SIZE)> broadcasted;

    std::vector<float> fqScales;

    // version based lazy evaluation, any parameter change increases parameterVersion
    // and postOpDataVersion will be compared with it to see if an update is required
    // when it was being actually used.
    size_t parameterVersion = 1lu;
    size_t postOpDataVersion = 0lu;
    size_t legacyPostOpDataVersion = 0lu;

    bool isInputLowBroadcasted = false;
    bool isInputHighBroadcasted = false;
    bool isOutputLowBroadcasted = false;
    bool isOutputHighBroadcasted = false;

    size_t currentAxisSize = 0;
    size_t axis = 0;

    ov::element::Type inputPrecision = ov::element::f32;
    ov::element::Type outputPrecision = ov::element::f32;

    BroadcastingPolicy broadcastingPolicy;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
