// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNExperimentalDetectronPriorGridGeneratorNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronPriorGridGeneratorNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    // Inputs:
    //      priors, shape [n, 4]
    //      [feature_map], shape [b, c, h, w]
    //      [im_data], shape [b, 3, im_h, im_w]
    // Outputs:
    //      priors_grid, shape [m, 4]

    const int INPUT_PRIORS {0};
    const int INPUT_FEATUREMAP {1};
    const int INPUT_IMAGE {2};

    const int OUTPUT_ROIS {0};

    int grid_w_;
    int grid_h_;
    float stride_w_;
    float stride_h_;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
