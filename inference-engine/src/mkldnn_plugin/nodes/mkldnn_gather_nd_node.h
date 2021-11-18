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

class MKLDNNGatherNDNode : public MKLDNNNode {
public:
    MKLDNNGatherNDNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
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
        size_t dstSize = 0lu;
        size_t sliceRank = 0lu;

        VectorDims srcDims;
        VectorDims srcStrides;
    } attrs;

    struct GatherNDExecutor {
        GatherNDExecutor(const GatherNDAttributes& attrs);
        ~GatherNDExecutor() = default;
        void exec(const uint8_t* srcData, const int32_t* indices, uint8_t* dstData);

    private:
        size_t batchSize = 1lu;
        size_t cycles = 1lu;
        size_t dataLength = 1lu;

        size_t srcBatchStride = 1lu;
        size_t idxBatchStride = 1lu;
        size_t dstBatchStride = 1lu;
        VectorDims srcShifts;

        GatherNDAttributes attrs;
    };

    static constexpr size_t GATHERND_DATA = 0lu;
    static constexpr size_t GATHERND_INDEXES = 1lu;

    using executorPtr = std::shared_ptr<GatherNDExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace MKLDNNPlugin
