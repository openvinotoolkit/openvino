// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../transpose.hpp"

namespace ov {
namespace intel_cpu {

template <typename T>
static void transpose_to_0312(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_04123(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr);
template<typename T>
static void transpose_to_051234(const int MB, const MemoryCPtr& srcMemPtr, MemoryPtr& dstMemPtr);

struct TransposeContext {
    MemoryCPtr srcMemPtr;
    MemoryPtr dstMemPtr;
    int MB;
};

template<typename T>
struct TransposeOptimizedEmitter {
    void operator()(TransposeContext& ctx) {
        switch (ctx.srcMemPtr->getStaticDims().size()) {
            case 4:
                transpose_to_0312<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                break;
            case 5:
                transpose_to_04123<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                break;
            case 6:
                transpose_to_051234<T>(ctx.MB, ctx.srcMemPtr, ctx.dstMemPtr);
                break;
            default:
                IE_THROW() << "Transpose supports optimized execution with only 4D, 5D and 6D shapes";
        }
    }
};

class RefTransposeExecutor : public TransposeExecutor {
public:
    explicit RefTransposeExecutor(const ExecutorContext::CPtr context);
    bool init(const TransposeParams &transposeParams,
              const std::vector<MemoryDescPtr> &srcDescs,
              const std::vector<MemoryDescPtr> &dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr> &src, const std::vector<MemoryPtr> &dst, const int MB) override;
    impl_desc_type getImplType() const override { return implType; }
private:
    impl_desc_type implType = impl_desc_type::ref;
};

class RefTransposeExecutorBuilder : public TransposeExecutorBuilder {
public:
    bool isSupported(const TransposeParams& transposeParams,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        return true;
    }

    TransposeExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<RefTransposeExecutor>(context);
    }
};

} // namespace intel_cpu
} // namespace ov