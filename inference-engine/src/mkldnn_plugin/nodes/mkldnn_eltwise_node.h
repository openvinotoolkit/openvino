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

enum EltwiseOpType {
    Add = 0,
    Multiply,
    Subtract,
    Divide,
    FloorMod,
    Mod,
    Maximum,
    Minimum,
    SquaredDifference,
    PowerDynamic,
    PowerStatic,
    MulAdd,

    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    LogicalAnd,
    LogicalOr,
    LogicalXor,
    LogicalNot,

    Relu,
    Gelu,
    Elu,
    Tanh,
    Logistic,
    Square,
    Abs,
    Sqrt,
    Linear,
    BoundedRelu,
    SoftRelu,
    Relu6,
    Exp,
    Clamp,
    Swish,
    Prelu,
    Mish,
    Hswish,
    Hsigmoid,
    Round
};

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

    jit_eltwise_params jep_;
    MKLDNNEltwiseNode& eltwiseNode;
};

class MKLDNNEltwiseNode : public MKLDNNNode {
public:
    MKLDNNEltwiseNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNEltwiseNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void selectOptimalPrimitiveDescriptor() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override;

    bool isSum();
    bool isWithBroadcast();

    bool canFuse(const MKLDNNNodePtr& node) const;

    size_t getOpInputsNum() const;
    EltwiseOpType getOpType() const { return eltwiseOp; }
    mkldnn::algorithm getAlgorithm() const { return eltwiseAlgorithm; }

    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }

    void appendPostOps(mkldnn::post_ops& ops) override;

    InferenceEngine::Precision getRuntimePrecision() const override;

private:
    void init() override;

    EltwiseOpType eltwiseOp = Add;
    mkldnn::algorithm eltwiseAlgorithm = mkldnn::algorithm_undef;

    std::shared_ptr<jit_uni_eltwise_kernel> eltwise_kernel = nullptr;
    jit_eltwise_params jep = {};

    int optimalTensorRank = 6;
    bool canUseOptimizedImpl = false;
    bool isDynBatchEnabled = false;
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

    inline void executeOptimized6D(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);
    inline void executeOptimizedGeneric(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);
    inline void executeReference(const std::vector<const uint8_t *>& src_ptrs, uint8_t *dst_ptr);

    void offset_out_calc(std::vector<size_t>& offset, std::vector<size_t>& dims);
    void offset_in_calc(std::vector<size_t>& offset, std::vector<size_t>& dims_in, std::vector<size_t>& dims_out);

    static InferenceEngine::details::caseless_map<std::string,
        std::function<void(InferenceEngine::GenericLayer*, EltwiseOpType&, mkldnn::algorithm&, float&, float&)>> initializers;
};

}  // namespace MKLDNNPlugin

