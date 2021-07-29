// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNExperimentalDetectronGenerateProposalsSingleImageNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronGenerateProposalsSingleImageNode(const std::shared_ptr<ngraph::Node>& op,
        const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      rois, shape [n, 4]
    //      rois_probs, shape [n]
    // Outputs:
    //      top_rois, shape [max_rois, 4]

    const int INPUT_IM_INFO {0};
    const int INPUT_ANCHORS {1};
    const int INPUT_DELTAS {2};
    const int INPUT_SCORES {3};
    const int OUTPUT_ROIS {0};
    const int OUTPUT_SCORES {1};

    float min_size_;
    int pre_nms_topn_;
    int post_nms_topn_;
    float nms_thresh_;
    float coordinates_offset;

    std::vector<int> roi_indices_;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
