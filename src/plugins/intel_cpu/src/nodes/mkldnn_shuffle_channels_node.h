// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include <memory>
#include <vector>
#include "common/permute_kernel.h"

namespace MKLDNNPlugin {

class MKLDNNShuffleChannelsNode : public MKLDNNNode {
public:
    MKLDNNShuffleChannelsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);
    ~MKLDNNShuffleChannelsNode() override = default;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void prepareParams() override;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    struct ShuffleChannelsAttributes {
        LayoutType layoutType;
        int dataRank = 0;
        int axis = 0;
        int spatialRank = 0;
        size_t group = 0lu;
        size_t dataSize = 1lu;
    } attrs;

    struct ShuffleChannelsExecutor final {
        ShuffleChannelsExecutor(const ShuffleChannelsAttributes& attrs, const VectorDims& srcDims, const VectorDims& srcBlockedDims);
        void exec(const uint8_t* srcData, uint8_t* dstData, const int MB);
        ~ShuffleChannelsExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel = nullptr;
    };
    using executorPtr = std::shared_ptr<ShuffleChannelsExecutor>;
    executorPtr execPtr = nullptr;

    bool supportDynamicBatch = false;
};

}  // namespace MKLDNNPlugin
