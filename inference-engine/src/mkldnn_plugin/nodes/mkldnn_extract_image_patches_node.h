// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

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

class MKLDNNExtractImagePatchesNode : public MKLDNNNode {
public:
    MKLDNNExtractImagePatchesNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

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
    InferenceEngine::Precision precision;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
