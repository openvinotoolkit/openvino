// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNDNode : public MKLDNNNode {
public:
    MKLDNNGatherNDNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;
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
        void exec(const MKLDNNMemoryPtr& srcMemPtr, const MKLDNNMemoryPtr& idxMemPtr, MKLDNNMemoryPtr& dstMemPtr);

    private:
        template <typename dataType>
        void gatherElementwise(const MKLDNNMemoryPtr& srcMemPtr, const MKLDNNMemoryPtr& idxMemPtr, MKLDNNMemoryPtr& dstMemPtr);
        void gatherBlocks(const MKLDNNMemoryPtr& srcMemPtr, const MKLDNNMemoryPtr& idxMemPtr, MKLDNNMemoryPtr& dstMemPtr);

        size_t batchSize = 1lu;
        size_t cycles = 1lu;
        size_t dataLength = 1lu;
        size_t sliceRank = 0lu;
        size_t workAmount = 0lu;
        size_t dataSize = 1lu;

        size_t srcBatchStride = 1lu;
        size_t idxBatchStride = 1lu;
        size_t dstBatchStride = 1lu;
        VectorDims srcShifts;

        struct GatherNDContext {
            GatherNDExecutor* executor;
            const MKLDNNMemoryPtr srcMemPtr;
            const MKLDNNMemoryPtr idxMemPtr;
            MKLDNNMemoryPtr dstMemPtr;
        };

        template<typename T>
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

}  // namespace MKLDNNPlugin
