// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

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
    void (*ker_)(const jit_extract_image_patches_args*);
    void operator()(const jit_extract_image_patches_args* args) {
        assert(ker_);
        ker_(args);
    }
    jit_extract_image_patches_params jpp;
    virtual void create_ker() = 0;
    explicit jit_uni_extract_image_patches_kernel(jit_extract_image_patches_params jpp) : ker_(nullptr), jpp(jpp) {}
    virtual ~jit_uni_extract_image_patches_kernel() {}
};

class ExtractImagePatches : public Node {
public:
    ExtractImagePatches(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;
    enum class ExtImgPatcherPadType { VALID, SAME_LOWER, SAME_UPPER };

private:
    std::vector<size_t> _ksizes;
    std::vector<size_t> _strides;
    std::vector<size_t> _rates;
    static const std::set<size_t> _supported_precisions_sizes;
    ExtImgPatcherPadType _auto_pad;

    struct ExtractImagePatchesExecutor {
        ExtractImagePatchesExecutor() = default;
        virtual void exec(void* src, void* dst, const VectorDims& istrides, const VectorDims& ostrides) = 0;
        jit_extract_image_patches_params fillJpp(const VectorDims& inDims,
                                                 const VectorDims& outDims,
                                                 const VectorDims& kSizes,
                                                 const VectorDims& strides,
                                                 const VectorDims& rates,
                                                 const ExtImgPatcherPadType& padType,
                                                 const size_t prcSize);
        virtual ~ExtractImagePatchesExecutor() = default;

    protected:
        size_t IC = 0;
        size_t IH = 0;
        size_t OB = 0;
        size_t RH = 0;
        size_t RW = 0;
        size_t PT = 0;
        size_t PL = 0;
    };

    using executorPtr = std::shared_ptr<ExtractImagePatchesExecutor>;
    executorPtr execPtr = nullptr;

    struct ExtractImagePatchesJitExecutor : public ExtractImagePatchesExecutor {
        ExtractImagePatchesJitExecutor(const VectorDims& inDims,
                                       const VectorDims& outDims,
                                       const VectorDims& kSizes,
                                       const VectorDims& strides,
                                       const VectorDims& rates,
                                       const ExtImgPatcherPadType& padType,
                                       const size_t prcSize);
        void exec(void* src, void* dst, const VectorDims& istrides, const VectorDims& ostrides) override;
        void executeOptimizedGeneric(void* src,
                                     void* dst,
                                     const VectorDims& istrides,
                                     const VectorDims& ostrides) const;

    private:
        std::unique_ptr<jit_uni_extract_image_patches_kernel> pKernel;
    };

    struct ExtractImagePatchesRefExecutor : public ExtractImagePatchesExecutor {
        ExtractImagePatchesRefExecutor(const VectorDims& inDims,
                                       const VectorDims& outDims,
                                       const VectorDims& kSizes,
                                       const VectorDims& strides,
                                       const VectorDims& rates,
                                       const ExtImgPatcherPadType& padType,
                                       const size_t prcSize);
        void exec(void* src, void* dst, const VectorDims& istrides, const VectorDims& ostrides) override;
        void executeReference(void* src, void* dst, const VectorDims& istrides, const VectorDims& ostrides) const;

    private:
        jit_extract_image_patches_params jpp;
    };
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
