// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <memory>

namespace MKLDNNPlugin {

struct jit_permute_conf_t {
    uint32_t ndims;
    InferenceEngine::SizeVector dst_block_dims;
    InferenceEngine::SizeVector src_strides;
    InferenceEngine::SizeVector dst_strides;
    int n;
    int data_size;

    bool supported_dynamic_batch = false;
};

struct jit_args_permute {
    const void* src;
    const void* dst;
};

struct jit_uni_permute_kernel {
    void (*ker_)(const jit_args_permute *);

    void operator()(const jit_args_permute *args) { assert(ker_); ker_(args); }

    jit_permute_conf_t jpp;

    explicit jit_uni_permute_kernel(jit_permute_conf_t jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_permute_kernel() {}
};

class MKLDNNPermuteNode : public MKLDNNNode {
public:
    MKLDNNPermuteNode(const InferenceEngine::CNNLayerPtr& layer, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNPermuteNode() override = default;

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;
    bool canBeInPlace() const override {
        return false;
    }

private:
    InferenceEngine::SizeVector order;
    InferenceEngine::Precision prec;

    typedef std::function<void(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> permuteImpl;
    typedef std::function<bool(int MB, MKLDNNMemoryPtr& srcMemPtr, MKLDNNMemoryPtr& dstMemPtr)> isApplicable;
    struct PermuteImpl {
        PermuteImpl(permuteImpl f0, isApplicable f1): execute(std::move(f0)), isValidParams(std::move(f1)) {}

        permuteImpl execute;
        isApplicable isValidParams;
    };

    static const std::multimap<InferenceEngine::SizeVector, PermuteImpl> OptimizedCases;
    std::shared_ptr<jit_uni_permute_kernel> permute_kernel;
};

}  // namespace MKLDNNPlugin

