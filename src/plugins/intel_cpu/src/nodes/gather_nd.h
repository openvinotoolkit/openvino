// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class GatherND : public Node {
public:
    GatherND(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(const dnnl::stream& strm) override;
    void prepareParams() override;

private:
    struct GatherNDAttributes {
        size_t batchDims = 0LU;
        size_t dataSize = 1LU;
        size_t dstElementCount = 0LU;
        size_t sliceRank = 0LU;

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
        int32_t HandleNegativeIndices(const int32_t* indices, size_t idx) const;

        size_t batchSize = 1LU;
        size_t dataSize = 1LU;
        size_t sliceRank = 0LU;
        size_t dataLength = 1LU;
        size_t cycles = 1LU;
        size_t workAmount = 0LU;

        size_t srcBatchStride = 1LU;
        size_t idxBatchStride = 1LU;
        size_t dstBatchStride = 1LU;
        VectorDims srcShifts;

        size_t batchDims = 0LU;
        VectorDims srcDims;

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

    static constexpr size_t GATHERND_DATA = 0LU;
    static constexpr size_t GATHERND_INDEXES = 1LU;

    using executorPtr = std::shared_ptr<GatherNDExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace ov::intel_cpu::node
