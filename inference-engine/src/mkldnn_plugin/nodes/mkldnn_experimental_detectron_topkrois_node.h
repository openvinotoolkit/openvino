// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNExperimentalDetectronTopKROIsNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronTopKROIsNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

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

    const int INPUT_ROIS {0};
    const int INPUT_PROBS {1};

    const int OUTPUT_ROIS {0};
    int max_rois_num_;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
