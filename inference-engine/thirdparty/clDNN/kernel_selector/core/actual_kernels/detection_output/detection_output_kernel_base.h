// Copyright (c) 2018-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


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
        float nms_threshold;
        float eta;
        float confidence_threshold;
    };

    DedicatedParams detectOutParams;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// detection_output_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct detection_output_optional_params : optional_params {
    detection_output_optional_params() : optional_params(KernelType::DETECTION_OUTPUT) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// DetectionOutputKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class DetectionOutputKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL ::KernelBaseOpenCL;
    virtual ~DetectionOutputKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const detection_output_params& params) const;
    virtual DispatchData SetDefault(const detection_output_params& params) const;
};
}  // namespace kernel_selector
