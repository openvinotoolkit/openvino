// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <cmath>
#include <mkldnn_extension_utils.h>
#include <ie_precision.hpp>
#include "ie_parallel.hpp"

namespace MKLDNNPlugin {

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

    void operator()(const jit_args_permute *args) { assert(ker_); ker_(args); }

    jit_permute_config_params jcp;

    explicit jit_uni_permute_kernel(jit_permute_config_params jcp_) : ker_(nullptr), jcp(jcp_) {}
    virtual ~jit_uni_permute_kernel() {}
};

class PermuteUtils {
protected:
    void prepareConfigParams();
    void optimizedExecute(const uint8_t* src_data, uint8_t* dst_data);

    inline uint8_t* getDataPtr(const MKLDNNPlugin::MKLDNNMemory& memoryPtr) const {
        return reinterpret_cast<uint8_t*>(memoryPtr.GetData()) + memoryPtr.GetDescriptor().data.layout_desc.blocking.offset_padding *
               MKLDNNExtensionUtils::sizeOfDataType(mkldnn::memory::data_type(memoryPtr.GetDescriptor().data.data_type));
    }

    InferenceEngine::SizeVector order;
    std::shared_ptr<jit_uni_permute_kernel> permute_kernel;

    struct {
        InferenceEngine::SizeVector src_dims;
        InferenceEngine::SizeVector src_block_dims;
        InferenceEngine::SizeVector src_block_order;
        InferenceEngine::SizeVector src_block_strides;

        InferenceEngine::SizeVector dst_dims;
        InferenceEngine::SizeVector dst_block_dims;
        InferenceEngine::SizeVector dst_block_order;
        InferenceEngine::SizeVector dst_block_strides;

        size_t data_size;
    } optimizedParams;
};

}  // namespace MKLDNNPlugin
