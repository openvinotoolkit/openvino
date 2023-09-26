// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>
#include <string>
#include <vector>
#include <memory>
#include <caseless.hpp>
#include "executors/eltwise_list.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

#define MAX_ELTWISE_INPUTS 7
#define MAX_ELTWISE_DIM_RANK 12

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    VectorDims dims;
    VectorDims src_offsets[MAX_ELTWISE_INPUTS];
    VectorDims dst_offsets;
    VectorDims oc_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;

    size_t work_amount;
    bool use_runtime_ptrs;
};

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
    //ptr to array of post op inputs pointers (flat list)
    const void** post_op_data;

    // shape agnostic kernel
    size_t work_amount;
    const void *src_offsets[MAX_ELTWISE_INPUTS];
    const void *dst_offsets;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

class Eltwise;

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep) : ker_(nullptr), jep_(jep) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
};

enum class EltwiseImplType {
    reference = 0,
    optimized = 1,
    optimizedShapeAgnostic = 2
};

class Eltwise : public Node {
public:
    struct EltwiseData {
        Algorithm algo;
        dnnl::algorithm onednnAlgorithm;
        float alpha;
        float beta;
        float gamma;

        bool operator==(const EltwiseData& rhs) const noexcept;
    };

    class IEltwiseExecutor {
    public:
        IEltwiseExecutor() = default;
        virtual void exec(const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) = 0;
        virtual size_t getBatchDimIdx() const = 0;
        virtual const VectorDims& getOutDims() const = 0;
        virtual ~IEltwiseExecutor() = default;
    };

    using executorPtr = std::shared_ptr<IEltwiseExecutor>;

public:
    Eltwise(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuse(const NodePtr& node) const override;
    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::unordered_map<int, MemoryPtr>& postOpsMem, const int channelAxis = 1) override;
    void appendPostOps(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<const void*>& postOpsMem, const int channelAxis = 1) override;
    bool appendAttrPostOps(DnnlPostOpsComposer& dnnlpoc, bool isLastPostOp, dnnl::memory::data_type outDataType, bool allowBinary = true);
    void fuseInto(NodePtr& parentNode) override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getGamma() const { return gamma; }

    dnnl::algorithm getOneDnnAlgorithm() const { return onednnAlgorithm; }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const { return specialConvolutionAddFusing; }

    bool needPrepareParams() const override;
    void prepareParams() override;
    void createPrimitive() override;

    void executeDynamicImpl(dnnl::stream strm) override;

    enum BroadcastingPolicy {
        PerChannel,
        PerTensor,
        Undefined,
    };

    BroadcastingPolicy getBroadcastingPolicy() const { return broadcastingPolicy; }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    executorPtr execPtr = nullptr;
    BroadcastingPolicy broadcastingPolicy;

    dnnl::algorithm onednnAlgorithm = dnnl::algorithm::undef;

    EltwiseImplType implType = EltwiseImplType::reference;
    std::vector<bool> broadcastPolicy;
    bool specialConvolutionAddFusing = false;
    size_t inputNum = 0;
    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;

    std::vector<InferenceEngine::Precision> inpPrc;
    InferenceEngine::Precision outPrc;

    // blocked dims for which kernel compiled and params prepared
    std::vector<VectorDims> currentInBlkDims = {};

    // shape agnostic kernel
    struct {
        VectorDims outDims;
        std::vector<VectorDims> inOffsets;
        VectorDims outOffsets;
    } execParams;

    float alpha = 0;
    float beta = 0;
    float gamma = 0;

    std::vector<float> scales = {};
    std::vector<float> shifts = {};
    MemoryPtr scalesMemory;
    MemoryPtr shiftsMemory;

    std::vector<float> depthwiseData = {};
    MemoryPtr depthwiseMemory;
    size_t depthwiseDataSize = 0;

    std::vector<MemoryPtr> memPtrs = {};
    std::vector<const void*> fqDataPtrs;

    using Initializer = std::function<void(const std::shared_ptr<ngraph::Node>&, Eltwise& node)>;
    static const std::map<const ngraph::DiscreteTypeInfo, Initializer>& getInitializers();

    static BroadcastingPolicy determineBroadcastingPolicy(const std::shared_ptr<ngraph::Node>& op);

    size_t getOpInputsNum() const;

    template <typename T>
    void appendPostOpsImpl(dnnl::post_ops& ops, const VectorDims &postOpDims, std::vector<T>& postOpsMem, const int channelAxis = 1);

    void appendMemory(const std::vector<float> &data, MemoryPtr &memPtr, std::vector<MemoryPtr>& postOpsMem);
    void appendMemory(const std::vector<float> &data, MemoryPtr &memPtr, std::vector<const void*>& postOpsMem);

    bool canUseAclExecutor = false;
    EltwiseAttrs eltwiseAttrs;
    std::shared_ptr<EltwiseExecutor> aclExecPtr = nullptr;
};

class eltwise_precision_helper {
public:
    static InferenceEngine::Precision get_precision(const size_t inputs_number,
                                                    const InferenceEngine::Precision (&src_prc)[MAX_ELTWISE_INPUTS],
                                                    const std::vector<Eltwise::EltwiseData>& eltwise_data);

private:
    static std::set<std::vector<element::Type>> get_supported_precisions(const Algorithm& algo);
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov