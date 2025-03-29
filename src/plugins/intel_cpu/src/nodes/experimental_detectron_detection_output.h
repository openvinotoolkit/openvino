// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {
namespace node {

class ExperimentalDetectronDetectionOutput : public Node {
public:
    ExperimentalDetectronDetectionOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override{};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    bool created() const override;

    bool needShapeInfer() const override;
    bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    const int INPUT_ROIS{0};
    const int INPUT_DELTAS{1};
    const int INPUT_SCORES{2};
    const int INPUT_IM_INFO{3};

    const int OUTPUT_BOXES{0};
    const int OUTPUT_CLASSES{1};
    const int OUTPUT_SCORES{2};

    float score_threshold_;
    float nms_threshold_;
    float max_delta_log_wh_;
    int classes_num_;
    int max_detections_per_class_;
    int max_detections_per_image_;
    bool class_agnostic_box_regression_;
    std::vector<float> deltas_weights_;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
