// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// detection_output_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct detection_output_params : public base_params {
    detection_output_params() : base_params(KernelType::DETECTION_OUTPUT), detectOutParams() {}

    struct DedicatedParams {
        uint32_t num_images;
        uint32_t num_classes;
        int32_t keep_top_k;
        int32_t top_k;
        int32_t background_label_id;
        int32_t code_type;
        int32_t conf_size_x;
        int32_t conf_size_y;
        int32_t conf_padding_x;
        int32_t conf_padding_y;
        int32_t elements_per_thread;
        int32_t input_width;
        int32_t input_heigh;
        int32_t prior_coordinates_offset;
        int32_t prior_info_size;
        bool prior_is_normalized;
        bool share_location;
        bool variance_encoded_in_target;
        bool decrease_label_id;
        bool clip_before_nms;
        bool clip_after_nms;
        float nms_threshold;
        float eta;
        float confidence_threshold;
    };

    DedicatedParams detectOutParams;

    ParamsKey GetParamsKey() const override {
        auto k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DetectionOutputKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DetectionOutputKernelRef: public KernelBaseOpenCL {
public:
    DetectionOutputKernelRef() : KernelBaseOpenCL("detection_output_gpu_ref") {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual JitConstants GetJitConstants(const detection_output_params& params) const;
    bool Validate(const Params& p) const override;
    void SetKernelArguments(const detection_output_params& params, clKernelData& kernel, size_t idx) const;
};
}  // namespace kernel_selector
