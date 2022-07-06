// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct experimental_detectron_detection_output_params : public base_params {
    experimental_detectron_detection_output_params()
        : base_params(KernelType::EXPERIMENTAL_DETECTRON_DETECTION_OUTPUT) {}

    float score_threshold{0.0f};
    float nms_threshold{0.0f};
    float max_delta_log_wh{0.0f};
    int num_classes{0};
    int post_nms_count{0};
    int max_detections_per_image{0};
    bool class_agnostic_box_regression{false};
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

private:
    JitConstants GetJitConstants(const experimental_detectron_detection_output_params& params) const;
    void PrepareKernelCommon(const experimental_detectron_detection_output_params& params,
                             const optional_params& options,
                             std::vector<size_t> gws,
                             const std::string& stage_name,
                             size_t stage_index,
                             clKernelData& kernel) const;
    void PrepareRefineBoxesKernel(const experimental_detectron_detection_output_params&,
                                  const optional_params&,
                                  clKernelData&) const;
    void PrepareNmsClassWiseKernel(const experimental_detectron_detection_output_params&,
                                   const optional_params&,
                                   clKernelData&) const;
    void PrepareTopKDetectionsKernel(const experimental_detectron_detection_output_params&,
                                     const optional_params&,
                                     clKernelData&) const;
    void PrepareCopyOutputKernel(const experimental_detectron_detection_output_params&,
                                 const optional_params&,
                                 clKernelData&) const;
};
}  // namespace kernel_selector
