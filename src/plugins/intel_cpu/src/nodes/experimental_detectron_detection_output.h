// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>

#include "graph_context.h"
#include "node.h"
#include "openvino/core/node.hpp"

namespace ov::intel_cpu::node {

class ExperimentalDetectronDetectionOutput : public Node {
public:
    ExperimentalDetectronDetectionOutput(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr& context);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void execute(const dnnl::stream& strm) override;
    [[nodiscard]] bool created() const override;

    [[nodiscard]] bool needShapeInfer() const override;
    [[nodiscard]] bool needPrepareParams() const override;
    void executeDynamicImpl(const dnnl::stream& strm) override {
        execute(strm);
    }
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

private:
    const int INPUT_ROIS{0};
    const int INPUT_DELTAS{1};
    const int INPUT_SCORES{2};

    const int OUTPUT_BOXES{0};
    const int OUTPUT_CLASSES{1};
    const int OUTPUT_SCORES{2};

    float score_threshold_;
    float nms_threshold_;
    float max_delta_log_wh_;
    int64_t classes_num_;
    int64_t max_detections_per_class_;
    int max_detections_per_image_;
    bool class_agnostic_box_regression_;
    std::vector<float> deltas_weights_;
};

}  // namespace ov::intel_cpu::node
