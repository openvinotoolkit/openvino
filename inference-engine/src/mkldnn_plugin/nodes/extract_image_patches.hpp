// Copyright (C) 2021 Intel Corporation
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
    int IW; // from input shape
    int IH; // ???
    int OH, OW; //from out shape
    int KH, KW; // kernel sizes
    int SW; // strides
    int SH; // ??
    int dtype_size; // byte size of the datatype
    int block_size; // num of dtype units in the supported vector instruction set
    InferenceEngine::Precision precision;
};

struct jit_eximpat_args {
    int64_t h_lo_pad;
    int64_t h_hi_pad;
    int64_t w_lo_pad;
    int64_t w_hi_pad;
    const void* src;
    void* dst; // const?
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
    template<typename T> void execute_fallback(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&) noexcept;
    void execute_optimized(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&) noexcept;


    std::vector<int64_t> _ksizes;
    std::vector<int64_t> _strides;
    std::vector<int64_t> _rates;
    std::vector<int64_t> _pads;
    std::string _auto_pad;

    std::shared_ptr<jit_uni_eximpat_kernel> eximpat_kernel;
    static const std::set<size_t> _supported_precisions_sizes;
    inline void set_pads(const std::string & pad_str, const std::vector<int64_t> & params);
};

const std::set<size_t> ExtractImagePatchesImpl::_supported_precisions_sizes = {1, 2, 4, 8};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
