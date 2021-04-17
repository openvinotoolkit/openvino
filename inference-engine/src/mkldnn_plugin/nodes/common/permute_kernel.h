// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <memory>

namespace MKLDNNPlugin {

namespace PermuteUtils {
struct PermuteParams {
    InferenceEngine::SizeVector src_dims;
    InferenceEngine::SizeVector src_block_dims;
    InferenceEngine::SizeVector src_block_order;
    InferenceEngine::SizeVector src_block_strides;

    InferenceEngine::SizeVector dst_dims;
    InferenceEngine::SizeVector dst_block_dims;
    InferenceEngine::SizeVector dst_block_order;
    InferenceEngine::SizeVector dst_block_strides;

    InferenceEngine::SizeVector order;
    size_t data_size;
};
}

struct jit_permute_config_params {
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

    void operator()(const jit_args_permute *args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_permute_kernel(jit_permute_config_params jcp_) : ker_(nullptr), jcp(jcp_) {}
    virtual ~jit_uni_permute_kernel() {}

    virtual void create_ker() = 0;

    jit_permute_config_params jcp;
};

class PermuteKernel {
public:
    PermuteKernel(const PermuteUtils::PermuteParams& params, const bool areDefault = true);

    bool isOptimized() {
        return permute_kernel != nullptr;
    }
    void optimizedExecute(const uint8_t* src_data, uint8_t* dst_data, const size_t mb);
    void execute(const uint8_t* src_data, uint8_t* dst_data, const size_t mb);

private:
    void prepareDefaultParams();
    void prepareParamsForOptimizedExecute();

    std::shared_ptr<jit_uni_permute_kernel> permute_kernel;
    PermuteUtils::PermuteParams params;
};

}  // namespace MKLDNNPlugin
