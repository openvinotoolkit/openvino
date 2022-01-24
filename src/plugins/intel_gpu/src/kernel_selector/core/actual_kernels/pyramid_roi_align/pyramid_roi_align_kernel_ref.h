// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "pyramid_roi_align_kernel_base.h"

namespace kernel_selector {
class PyramidROIAlignKernelRef : public PyramidROIAlignKernelBase {
public:
    PyramidROIAlignKernelRef() : PyramidROIAlignKernelBase("pyramid_roi_align_gpu_ref") {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    DispatchData SetDefault(const PyramidROIAlign_params& params) const override;
};
}  // namespace kernel_selector
