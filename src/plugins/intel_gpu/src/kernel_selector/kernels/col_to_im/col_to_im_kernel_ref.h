// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "col_to_im_kernel_base.h"

namespace kernel_selector {
class ColToImKernelRef : public ColToImKernelBase {
public:
    using Parent = ColToImKernelBase;

    ColToImKernelRef() : ColToImKernelBase("col_to_im_ref") {}
    virtual ~ColToImKernelRef() {}

    CommonDispatchData SetDefault(const col_to_im_params& params) const override;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    JitConstants GetJitConstants(const col_to_im_params& params) const override;
};
}  // namespace kernel_selector
