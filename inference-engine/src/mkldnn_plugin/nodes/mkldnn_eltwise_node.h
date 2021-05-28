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

struct jit_eltwise_params {
    size_t inputs_number;
    size_t input_size;

    InferenceEngine::Precision src_prc[MAX_ELTWISE_INPUTS];
    InferenceEngine::Precision dst_prc;

    std::vector<size_t> src_offsets[MAX_ELTWISE_INPUTS];
    std::vector<size_t> dst_offsets;

    size_t src_size[MAX_ELTWISE_INPUTS];
    size_t dst_size;
    size_t oc_size;
};

struct jit_eltwise_call_args {
    const void *src_ptr[MAX_ELTWISE_INPUTS];
    void *dst;

    size_t work_amount;
    size_t oc_off;
};

class MKLDNNEltwiseNode;

struct jit_uni_eltwise_kernel {
    void (*ker_)(const jit_eltwise_call_args *);

    void operator()(const jit_eltwise_call_args *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_eltwise_kernel(jit_eltwise_params jep, MKLDNNEltwiseNode& node) : ker_(nullptr), jep_(jep), eltwiseNode(node) {}
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
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;
    bool canFuse(const MKLDNNNodePtr& node) const override;
    void appendPostOps(mkldnn::post_ops& ops) override;
    void fuseInto(MKLDNNNodePtr& parentNode) override;
    InferenceEngine::Precision getRuntimePrecision() const override;

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getGamma() const { return gamma; }
    mkldnn::algorithm getMKLDNNAlgorithm() const { return mkldnnAlgorithm; }

    bool isWithBroadcast();
    bool isSpecialConvolutionAddFusing() const { return specialConvolutionAddFusing; }

private:
    mkldnn::algorithm mkldnnAlgorithm = mkldnn::algorithm::undef;

    std::shared_ptr<jit_uni_eltwise_kernel> eltwise_kernel = nullptr;
    jit_eltwise_params jep = {};

    int optimalTensorRank = 6;
    bool canUseOptimizedImpl = false;
    bool isDynBatchEnabled = false;
    bool specialConvolutionAddFusing = false;
    size_t batchDimIdx = 0;
    size_t tensorRank = 0;
    size_t fullWorkAmount = 0;
    size_t schedulerWorkAmount = 0;
    std::vector<std::vector<size_t>> dims_in = {};
    std::vector<std::vector<size_t>> offsets_in = {};
    std::vector<size_t> dims_out = {};
    std::vector<size_t> offsets_out = {};
    std::vector<ptrdiff_t> start_offset_in = {};
    ptrdiff_t start_offset_out = 0;
    std::vector<size_t> offsets_oc = {};

    float alpha = 0;
    float beta = 0;
    float gamma = 0;

    std::vector<float> scales = {};
    std::vector<float> shifts = {};

    static std::map<const ngraph::DiscreteTypeInfo, std::function<void(const std::shared_ptr<ngraph::Node>&, MKLDNNEltwiseNode& node)>> initializers;

    inline void executeOptimized6D(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);
    inline void executeOptimizedGeneric(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);
    inline void executeReference(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);

    void offset_out_calc(std::vector<size_t>& offset, std::vector<size_t>& dims);
    void offset_in_calc(std::vector<size_t>& offset, std::vector<size_t>& dims_in, std::vector<size_t>& dims_out);

    size_t getOpInputsNum() const;
};

}  // namespace MKLDNNPlugin

