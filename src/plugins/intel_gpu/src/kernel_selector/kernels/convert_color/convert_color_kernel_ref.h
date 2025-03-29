// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convert_color_kernel_base.h"

namespace kernel_selector {
class ConvertColorKernelRef : public ConvertColorKernelBase {
public:
    using Parent = ConvertColorKernelBase;
    ConvertColorKernelRef() : ConvertColorKernelBase("convert_color_ref") {}
    virtual ~ConvertColorKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
};
}  // namespace kernel_selector
