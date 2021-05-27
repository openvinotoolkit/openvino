// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "base.hpp"
#include <vector>
#include <set>
#include <cassert>

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {

struct jit_extract_image_patches_params {
    size_t IW;
    size_t OH, OW;
    size_t KH, KW;
    size_t SH, SW;
    size_t dtype_size;
    size_t block_size;
    bool need_padding;
};

struct jit_extract_image_patches_args {
    uint64_t h_lo_pad;
    uint64_t h_hi_pad;
    uint64_t w_lo_pad;
    uint64_t w_hi_pad;
    const void* src;
    void* dst;
};

struct jit_uni_extract_image_patches_kernel {
    void (*ker_)(const jit_extract_image_patches_args *);
    void operator()(const jit_extract_image_patches_args *args) { assert(ker_); ker_(args); }
    jit_extract_image_patches_params jpp;
    virtual void create_ker() = 0;
    explicit jit_uni_extract_image_patches_kernel(jit_extract_image_patches_params jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_extract_image_patches_kernel() {}
};


class ExtractImagePatchesImpl : public ExtLayerBase {
public:
    explicit ExtractImagePatchesImpl(const std::shared_ptr<ngraph::Node>& op);
    StatusCode execute(std::vector<Blob::Ptr>&, std::vector<Blob::Ptr>&, ResponseDesc*) noexcept override;
    bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    enum class ExtImgPatcherPadType {
        VALID,
        SAME_LOWER,
        SAME_UPPER
    };

    std::vector<size_t> _ksizes;
    std::vector<size_t> _strides;
    std::vector<size_t> _rates;
    size_t _pad_left;
    size_t _pad_top;
    std::shared_ptr<jit_uni_extract_image_patches_kernel> extract_image_patches_kernel;
    static const std::set<size_t> _supported_precisions_sizes;

    ExtImgPatcherPadType _auto_pad;

    std::string errorPrefix;
};

REG_FACTORY_FOR(ExtractImagePatchesImpl, ExtractImagePatches);

}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine
