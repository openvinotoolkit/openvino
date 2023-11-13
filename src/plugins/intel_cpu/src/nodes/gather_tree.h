// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

namespace ov {
namespace intel_cpu {
namespace node {

class GatherTree : public Node {
public:
    GatherTree(const std::shared_ptr<ngraph::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    struct GatherTreeExecutor {
        GatherTreeExecutor(const VectorDims& stepIdxDims,
                           const VectorDims& parentIdxDims,
                           const VectorDims& maxSeqLenDims,
                           const VectorDims& dstDims);
        ~GatherTreeExecutor() = default;

        template<typename DATA_T>
        void exec(const MemoryPtr& stepIdxMemPtr,
                  const MemoryPtr& parentIdxMemPtr,
                  const MemoryPtr& maxSeqLenMemPtr,
                  const MemoryPtr& endTokenMemPtr,
                  const MemoryPtr& dstMemPtr);

    private:
        const int32_t maxTime;
        const size_t batchSize;
        const size_t beamWidth;
        const size_t bbSize;
        const size_t parentIdxSize;
    };

    using executorPtr = std::shared_ptr<GatherTreeExecutor>;
    executorPtr execPtr = nullptr;

    static const size_t GATHER_TREE_STEP_IDX = 0;
    static const size_t GATHER_TREE_PARENT_IDX = 1;
    static const size_t GATHER_TREE_MAX_SEQ_LEN = 2;
    static const size_t GATHER_TREE_END_TOKEN = 3;

    ov::element::Type precision;

    std::string errorPrefix;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
