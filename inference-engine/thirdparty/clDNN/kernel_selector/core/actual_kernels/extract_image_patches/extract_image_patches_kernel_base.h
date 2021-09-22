// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    virtual JitConstants GetJitConstants(const extract_image_patches_params& params) const;
    DispatchData SetDefault(const extract_image_patches_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;

    ParamsKey GetSupportedKey() const override;
    bool Validate(const Params& p, const optional_params&) const override;
};
}  // namespace kernel_selector
