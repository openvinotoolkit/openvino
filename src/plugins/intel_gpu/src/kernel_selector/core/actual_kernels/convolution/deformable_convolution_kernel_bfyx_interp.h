// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_kernel_base.h"

namespace kernel_selector {

class DeformableConvolutionKernel_bfyx_interp : public KernelBaseOpenCL {
public:
    DeformableConvolutionKernel_bfyx_interp() : KernelBaseOpenCL("deformable_convolution_gpu_bfyx_interp") {}
    virtual ~DeformableConvolutionKernel_bfyx_interp() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;

protected:
    virtual CommonDispatchData SetDefault(const convolution_params& params) const;
    virtual JitConstants GetJitConstants(const convolution_params& params) const;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
