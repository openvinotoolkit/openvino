// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class GatherND : public Node {
public:
    GatherND(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    struct GatherNDAttributes {
        size_t batchDims = 0lu;
        size_t dataSize = 1lu;
        size_t dstElementCount = 0lu;
        size_t sliceRank = 0lu;

        VectorDims srcDims;
        VectorDims srcStrides;
    } attrs;

    struct GatherNDExecutor {
        GatherNDExecutor(const GatherNDAttributes& attrs);
        ~GatherNDExecutor() = default;
        void exec(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr);

    private:
        template <typename dataType>
        void gatherElementwise(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr);
        void gatherBlocks(const MemoryPtr& srcMemPtr, const MemoryPtr& idxMemPtr, const MemoryPtr& dstMemPtr);

        size_t batchSize = 1lu;
        size_t dataSize = 1lu;
        size_t sliceRank = 0lu;
        size_t dataLength = 1lu;
        size_t cycles = 1lu;
        size_t workAmount = 0lu;

        size_t srcBatchStride = 1lu;
        size_t idxBatchStride = 1lu;
        size_t dstBatchStride = 1lu;
        VectorDims srcShifts;

        struct GatherNDContext {
            GatherNDExecutor* executor;
            const MemoryPtr srcMemPtr;
            const MemoryPtr idxMemPtr;
            MemoryPtr dstMemPtr;
        };

        template <typename T>
        struct GatherNDEmitter {
            void operator()(GatherNDContext& ctx) {
                ctx.executor->gatherElementwise<T>(ctx.srcMemPtr, ctx.idxMemPtr, ctx.dstMemPtr);
            }
        };
    };

    static constexpr size_t GATHERND_DATA = 0lu;
    static constexpr size_t GATHERND_INDEXES = 1lu;

    using executorPtr = std::shared_ptr<GatherNDExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
