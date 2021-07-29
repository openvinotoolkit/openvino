// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNExperimentalDetectronROIFeatureExtractorNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronROIFeatureExtractorNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const int INPUT_ROIS {0};
    const int INPUT_FEATURES_START {1};

    const int OUTPUT_ROI_FEATURES {0};
    const int OUTPUT_ROIS {1};

    int output_dim_ = 0;
    int pooled_height_ = 0;
    int pooled_width_ = 0;
    std::vector<int64_t> pyramid_scales_;
    int sampling_ratio_ = 0;
    bool aligned_ = false;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
