// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <utility>

#include "node.h"

namespace ov {
namespace intel_cpu {

struct PermuteParams {
    VectorDims src_block_dims;
    VectorDims dst_block_dims;
    VectorDims src_block_order;
    VectorDims dst_block_order;
    VectorDims order;
    size_t data_size;

    size_t hash() const;
    bool operator==(const PermuteParams& rhs) const;
};

struct jit_permute_config_params {
    uint32_t ndims;
    VectorDims dst_block_dims;
    VectorDims src_strides;
    VectorDims dst_strides;
    int n;
    int data_size;

    bool supported_dynamic_batch = false;
};

struct jit_args_permute {
    const void* src;
    const void* dst;
};

struct jit_uni_permute_kernel {
    void (*ker_)(const jit_args_permute*);

    void operator()(const jit_args_permute* args) {
        assert(ker_);
        ker_(args);
    }

    explicit jit_uni_permute_kernel(jit_permute_config_params jcp_) : ker_(nullptr), jcp(std::move(jcp_)) {}
    virtual ~jit_uni_permute_kernel() {}

    virtual void create_ker() = 0;

    jit_permute_config_params jcp;
};

class PermuteKernel {
public:
    PermuteKernel(const PermuteParams& params);

    void execute(const uint8_t* src_data, uint8_t* dst_data);
    void execute(const uint8_t* src_data, uint8_t* dst_data, const int mb);
    const PermuteParams& getPermuteParams() const {
        return params;
    }

private:
    void optimizedExecute(const uint8_t* src_data, uint8_t* dst_data, const int mb);

    jit_permute_config_params jcp = {};
    std::shared_ptr<jit_uni_permute_kernel> permute_kernel;
    PermuteParams params;
};

}  // namespace intel_cpu
}  // namespace ov
