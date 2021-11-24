// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include <string>
#include "common/permute_kernel.h"

namespace MKLDNNPlugin {

class MKLDNNSpaceToDepthNode : public MKLDNNNode {
public:
    MKLDNNSpaceToDepthNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;
    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    void prepareParams() override;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;

private:
    enum Mode {
        BLOCKS_FIRST = 0,
        DEPTH_FIRST = 1
    };

    struct SpaceToDepthAttrs {
        LayoutType layoutType;
        Mode mode;
        size_t blockSize = 0lu;
        size_t blockStep = 1lu;
        size_t dataSize = 1lu;
        size_t nSpatialDims = 0lu;
    } attrs;

    struct SpaceToDepthExecutor final {
        SpaceToDepthExecutor(const SpaceToDepthAttrs& attrs,
                             const VectorDims& srcBlockedDims,
                             const VectorDims& dstBlockedDims);
        void exec(const uint8_t* srcData, uint8_t* dstData, const int MB);
        ~SpaceToDepthExecutor() = default;

    private:
        std::unique_ptr<PermuteKernel> permuteKernel;
    };
    using executorPtr = std::shared_ptr<SpaceToDepthExecutor>;
    executorPtr execPtr = nullptr;
};

}  // namespace MKLDNNPlugin
