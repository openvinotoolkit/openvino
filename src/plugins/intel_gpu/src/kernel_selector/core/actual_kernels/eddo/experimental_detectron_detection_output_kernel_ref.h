// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct experimental_detectron_detection_output_params : public base_params {
    experimental_detectron_detection_output_params()
        : base_params(KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT) {}

    float score_threshold;
    float nms_threshold;
    float max_delta_log_wh;
    int64_t num_classes;
    int64_t post_nms_count;
    size_t max_detections_per_image;
    bool class_agnostic_box_regression;
    std::vector<float> deltas_weights;
};

struct experimental_detectron_detection_output_optional_params : public optional_params {
    experimental_detectron_detection_output_optional_params()
        : optional_params(KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT) {}
};

class ExperimentalDetectronDetectionOutputKernelRef : public KernelBaseOpenCL {
public:
    ExperimentalDetectronDetectionOutputKernelRef() : KernelBaseOpenCL("experimental_detectron_detection_output_ref") {}

    ~ExperimentalDetectronDetectionOutputKernelRef() = default;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
