// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <memory>
#include <caseless.hpp>

namespace MKLDNNPlugin {

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
};

struct jit_eltwise_call_args_ptrs {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst_ptr;
};

struct jit_eltwise_call_args_indexes {
    size_t indexes[MAX_ELTWISE_DIM_RANK];
};

class MKLDNNEltwiseNode;

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args_ptrs*, const jit_eltwise_call_args_indexes*);

    void operator()(const jit_eltwise_call_args_ptrs* const_args, const jit_eltwise_call_args_indexes* indexes) {
        assert(ker_);
        ker_(const_args, indexes);
    }

    explicit jit_uni_eltwise_kernel(const jit_eltwise_params& jep, MKLDNNEltwiseNode& node) : ker_(nullptr), jep_(jep), eltwiseNode(node) {}
    virtual ~jit_uni_eltwise_kernel() {}

    virtual void create_ker() = 0;

    jit_eltwise_params jep_;
    MKLDNNEltwiseNode& eltwiseNode;
};

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    MKLDNNEltwiseNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void initOptimalPrimitiveDescriptor() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    void appendPostOps(mkldnn::post_ops& ops, const VectorDims &postOpDims, int align = -1) override;
    void appendBinPostOps(mkldnn::post_ops& ops, const VectorDims &postOpDims, std::vector<MKLDNNMemoryPtr>& binaryPostOpsMem) override;
    void fuseInto(MKLDNNNodePtr& parentNode) override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getGamma() const { return gamma; }
    MKLDNNMemoryPtr scalesMemory;
    MKLDNNMemoryPtr shiftsMemory;
    mkldnn::algorithm getMKLDNNAlgorithm() const { return mkldnnAlgorithm; }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const { return specialConvolutionAddFusing; }

    void createPrimitive() override;

    std::vector<VectorDims> shapeInfer() const override;
    bool needPrepareParams() const override;
    void prepareParams() override;

    void executeDynamicImpl(mkldnn::stream strm) override { execute(strm); }

    enum BroadcastingPolicy {
        PerChannel,
        PerTensor,
        Undefined,
    };

    BroadcastingPolicy getBroadcastingPolicy() const { return broadcastingPolicy; }

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;


private:
    struct EltwiseExecutor {
        EltwiseExecutor(size_t batch) : batchDimIdx(batch) {}
        virtual void exec(const MKLDNNEltwiseNode& node, const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) = 0;
        virtual const jit_eltwise_params& getJep() const = 0;
        virtual ~EltwiseExecutor() = default;

        size_t batchDimIdx = 0;
    };
    using executorPtr = std::shared_ptr<EltwiseExecutor>;
    executorPtr execPtr = nullptr;

    struct EltwiseJitExecutor : public EltwiseExecutor {
        EltwiseJitExecutor(const jit_eltwise_params &_jep, MKLDNNEltwiseNode& node, const size_t schedWA, const size_t batch);
        void exec(const MKLDNNEltwiseNode& node, const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) override;
        const jit_eltwise_params& getJep() const override;

        std::unique_ptr<jit_uni_eltwise_kernel> pKernel;
        size_t schedulerWorkAmount = 0;
    };

    struct EltwiseRefExecutor : public EltwiseExecutor {
        EltwiseRefExecutor(const jit_eltwise_params &_jep, const size_t fullWA, const size_t batch) : jep(_jep), fullWorkAmount(fullWA),
                                                                                                       EltwiseExecutor(batch) {}
        void exec(const MKLDNNEltwiseNode& node, const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out) override;
        const jit_eltwise_params& getJep() const override { return jep; }

        jit_eltwise_params jep;
        size_t fullWorkAmount = 0;
    };

    BroadcastingPolicy broadcastingPolicy;

    mkldnn::algorithm mkldnnAlgorithm = mkldnn::algorithm::undef;

    static const int optimalTensorRank = 6;
    bool canUseOptimizedImpl = false;
    bool isDynBatchEnabled = false;
    bool specialConvolutionAddFusing = false;
    size_t inputNum = 0;
    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;

    // blocked dims for which kernel compiled and params prepared
    std::vector<VectorDims> currentInBlkDims = {};

    float alpha = 0;
    float beta = 0;
    float gamma = 0;

    std::vector<float> scales = {};
    std::vector<float> shifts = {};
    std::vector<float> scalesBuffer = {};
    std::vector<float> shiftsBuffer = {};

    std::vector<MKLDNNMemoryPtr> memPtrs = {};

    using Initializer = std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNEltwiseNode& node)>;
    static const std::map<const ngraph::DiscreteTypeInfo, Initializer> initializers;

    static BroadcastingPolicy determineBroadcastingPolicy(const std::shared_ptr<ngraph::Node>& op);

    void executeOptimized6D(const std::unique_ptr<jit_uni_eltwise_kernel> &pKernel, const jit_eltwise_call_args_ptrs &args_ptrs,
                            const VectorDims &dims_out) const;
    void executeOptimizedGeneric(const std::unique_ptr<jit_uni_eltwise_kernel> &pKernel, const jit_eltwise_call_args_ptrs &args_ptrs,
                                 const VectorDims &dims_out, const size_t schedulerWorkAmount) const;
    void executeReference(const jit_eltwise_params &jep, const jit_eltwise_call_args_ptrs &args_ptrs, const VectorDims &dims_out,
                          const size_t fullWorkAmount) const;

    void offset_out_calc(VectorDims& offset, VectorDims& dims);
    void offset_in_calc(VectorDims& offset, VectorDims& dims_in, VectorDims& dims_out);

    size_t getOpInputsNum() const;
};

}  // namespace MKLDNNPlugin
