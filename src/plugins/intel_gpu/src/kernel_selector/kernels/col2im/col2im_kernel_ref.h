// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "col2im_kernel_base.h"

namespace kernel_selector {
class Col2ImKernelRef : public Col2ImKernelBase {
public:
    using Parent = Col2ImKernelBase;

    Col2ImKernelRef() : Col2ImKernelBase("col2im_ref") {}
    virtual ~Col2ImKernelRef() {}

    CommonDispatchData SetDefault(const col2im_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const col2im_params& params) const override;
};
}  // namespace kernel_selector
