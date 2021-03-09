/*
// Copyright (c) 2021 Intel Corporation
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
*/

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non_max_suppression_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct non_max_suppression_params : public base_params {
    non_max_suppression_params() : base_params(KernelType::NON_MAX_SUPPRESSION),
    box_encoding(0), sort_result_descending(true), has_num_select_per_class(false),
    has_iou_threshold(false), has_score_threshold(false), has_soft_nms_sigma(false) {}

    uint32_t box_encoding;  // 0(corner), 1(center)
    bool sort_result_descending;
    // clDNN primitive supports only i32 as output data type
    bool has_num_select_per_class;
    bool has_iou_threshold;
    bool has_score_threshold;
    bool has_soft_nms_sigma;

    virtual ParamsKey GetParamsKey() const {
        auto k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// non_max_suppression_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct non_max_suppression_optional_params : optional_params {
    non_max_suppression_optional_params() : optional_params(KernelType::NON_MAX_SUPPRESSION) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// NonMaxSuppressionKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class NonMaxSuppressionKernelRef : public KernelBaseOpenCL {
public:
    NonMaxSuppressionKernelRef() : KernelBaseOpenCL("non_max_suppression_gpu_ref") {}
    virtual ~NonMaxSuppressionKernelRef() {}

    using DispatchData = CommonDispatchData;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    virtual JitConstants GetJitConstants(const non_max_suppression_params& params) const;
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
