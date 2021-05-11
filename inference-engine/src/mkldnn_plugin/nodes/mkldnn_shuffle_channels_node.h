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

    void getSupportedDescriptors() override;
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    ngraph::Shape inShape_;
    int dataRank_;
    int axis_;
    size_t group_;
    size_t groupSize_;

    std::unique_ptr<PermuteKernel> permuteKernel_;
    bool supportDynamicBatch_;
};

}  // namespace MKLDNNPlugin
