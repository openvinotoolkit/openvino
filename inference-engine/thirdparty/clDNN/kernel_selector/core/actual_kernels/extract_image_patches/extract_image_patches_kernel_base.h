// Copyright (c) 2020 Intel Corporation
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

#include <vector>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract_image_patches_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct extract_image_patches_params : public base_params {
    extract_image_patches_params() : base_params(KernelType::EXTRACT_IMAGE_PATCHES) {}

    std::vector<unsigned int> sizes;
    std::vector<unsigned int> strides;
    std::vector<unsigned int> rates;
    std::string auto_pad;

    virtual ParamsKey GetParamsKey() const { return base_params::GetParamsKey(); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// extract_image_patches_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct extract_image_patches_optional_params : optional_params {
    extract_image_patches_optional_params() : optional_params(KernelType::EXTRACT_IMAGE_PATCHES) {}
};

class ExtractImagePatchesKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    using DispatchData = CommonDispatchData;
    virtual ~ExtractImagePatchesKernelBase() {}

protected:
    virtual ParamsKey GetSupportedKey() const override;
    virtual JitConstants GetJitConstants(const extract_image_patches_params& params) const;
    DispatchData SetDefault(const extract_image_patches_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&, float estimated_time) const;

    bool Validate(const Params& p, const optional_params&) const override;
};
}  // namespace kernel_selector
