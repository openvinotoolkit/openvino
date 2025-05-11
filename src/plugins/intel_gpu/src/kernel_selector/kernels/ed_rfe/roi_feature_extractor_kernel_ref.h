// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
struct experimental_detectron_roi_feature_extractor_params : public base_params {
    experimental_detectron_roi_feature_extractor_params() : base_params(KernelType::EXPERIMENTAL_DETECTRON_ROI_FEATURE_EXTRACTOR) {}

    int output_dim = 0;
    int pooled_height = 0;
    int pooled_width = 0;
    std::vector<int64_t> pyramid_scales;
    int sampling_ratio = 0;
    bool aligned = false;
    std::size_t number_of_inputs = 0;
};

class ExperimentalDetectronROIFeatureExtractorRef : public KernelBaseOpenCL {
public:
    ExperimentalDetectronROIFeatureExtractorRef() : KernelBaseOpenCL("experimental_detectron_roi_feature_extractor_ref") {}
    ~ExperimentalDetectronROIFeatureExtractorRef() = default;

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    virtual JitConstants GetJitConstants(const experimental_detectron_roi_feature_extractor_params& params) const;
};
}  // namespace kernel_selector
