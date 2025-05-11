// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "roi_pooling_kernel_base.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// PSROIPoolingKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class PSROIPoolingKernelRef : public ROIPoolingKernelBase {
public:
    PSROIPoolingKernelRef() : ROIPoolingKernelBase("roi_pooling_ps_ref") {}
    virtual ~PSROIPoolingKernelRef() {}

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const roi_pooling_params& params) const override;
};
}  // namespace kernel_selector
