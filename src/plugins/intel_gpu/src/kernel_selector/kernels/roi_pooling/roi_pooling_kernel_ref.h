// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "roi_pooling_kernel_base.h"

namespace kernel_selector {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ROIPoolingKernelRef
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class ROIPoolingKernelRef : public ROIPoolingKernelBase {
public:
    ROIPoolingKernelRef() : ROIPoolingKernelBase("roi_pooling_ref") {}
    virtual ~ROIPoolingKernelRef() {}

    using DispatchData = CommonDispatchData;

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
};
}  // namespace kernel_selector
