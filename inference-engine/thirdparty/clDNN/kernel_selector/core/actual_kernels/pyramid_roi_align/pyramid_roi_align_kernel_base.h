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
// PyramidROIAlign_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PyramidROIAlign_params : public base_params {
    PyramidROIAlign_params() : base_params(KernelType::PYRAMID_ROI_ALIGN),
    image_size_x(1), image_size_y(1), sampling_ratio_x(1), sampling_ratio_y(1),
    pyramid_starting_level(0) {}

    int image_size_x;
    int image_size_y;
    int sampling_ratio_x;
    int sampling_ratio_y;
    int pyramid_starting_level;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// index_select_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct PyramidROIAlign_optional_params : optional_params {
    PyramidROIAlign_optional_params() : optional_params(KernelType::PYRAMID_ROI_ALIGN) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PyramidROIAlignKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PyramidROIAlignKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~PyramidROIAlignKernelBase() {}

    using DispatchData = CommonDispatchData;

protected:
    JitConstants GetJitConstants(const PyramidROIAlign_params& params) const;
    virtual DispatchData SetDefault(const PyramidROIAlign_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
