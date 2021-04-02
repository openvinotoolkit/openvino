// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "base.hpp"
#include "caseless.hpp"
#include "ie_parallel.hpp"

#include <string>
#include <vector>
#include <set>
#include <cmath>
// algo is for debug purposes
#include <algorithm>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct jit_eximpat_params {
    int IW;
    int OH, OW;
    int KH, KW;
    int SH, SW;
    int dtype_size;
    int block_size;
};

struct jit_eximpat_args {
    int64_t h_lo_pad;
    int64_t h_hi_pad;
    int64_t w_lo_pad;
    int64_t w_hi_pad;
    const void* src;
    void* dst;
};

struct jit_uni_eximpat_kernel {
    void (*ker_)(const jit_eximpat_args *);
    void operator()(const jit_eximpat_args *args) { assert(ker_); ker_(args); }
    jit_eximpat_params jpp;
    virtual void create_ker() = 0;
    explicit jit_uni_eximpat_kernel(jit_eximpat_params jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_eximpat_kernel() {}
};

using details::CaselessEq;

class ExtractImagePatchesImpl : public ExtLayerBase {
public:
    explicit ExtractImagePatchesImpl(const CNNLayer*);
    StatusCode execute(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&, ResponseDesc*) noexcept override;

private:
    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    int64_t _pad_left;
    int64_t _pad_top;
    std::shared_ptr<jit_uni_eximpat_kernel> eximpat_kernel;
    static const std::set<size_t> _supported_precisions_sizes;
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
