// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>
#include "proposal_imp.hpp"

using proposal_conf = InferenceEngine::Extensions::Cpu::proposal_conf;

namespace MKLDNNPlugin {

class MKLDNNProposalNode : public MKLDNNNode {
public:
    MKLDNNProposalNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    bool needPrepareParams() const override { return false; };
    void executeDynamicImpl(mkldnn::stream strm) override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const size_t PROBABILITIES_IN_IDX = 0lu;
    const size_t ANCHORS_IN_IDX = 1lu;
    const size_t IMG_INFO_IN_IDX = 2lu;
    const size_t ROI_OUT_IDX = 0lu;
    const size_t PROBABILITIES_OUT_IDX = 1lu;

    proposal_conf conf;
    std::vector<float> anchors;
    std::vector<int> roi_indices;
    bool store_prob;  // store blob with proposal probabilities
};

}  // namespace MKLDNNPlugin
