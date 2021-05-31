// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <mkldnn_node.h>

namespace MKLDNNPlugin {

class MKLDNNExperimentalDetectronDetectionOutputNode : public MKLDNNNode {
public:
    MKLDNNExperimentalDetectronDetectionOutputNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override {};
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<ngraph::Node>& op, std::string& errorMessage) noexcept;

private:
    const int INPUT_ROIS {0};
    const int INPUT_DELTAS {1};
    const int INPUT_SCORES {2};
    const int INPUT_IM_INFO {3};

    const int OUTPUT_BOXES {0};
    const int OUTPUT_CLASSES {1};
    const int OUTPUT_SCORES {2};

    float score_threshold_;
    float nms_threshold_;
    float max_delta_log_wh_;
    int classes_num_;
    int max_detections_per_class_;
    int max_detections_per_image_;
    bool class_agnostic_box_regression_;
    std::vector<float> deltas_weights_;

    std::string errorPrefix;
};

}  // namespace MKLDNNPlugin
