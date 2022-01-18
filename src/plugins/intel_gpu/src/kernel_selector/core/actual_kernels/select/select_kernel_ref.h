// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "select_kernel_base.h"

namespace kernel_selector {
class SelectKernelRef : public SelectKernelBase {
public:
    SelectKernelRef() : SelectKernelBase("select_gpu_ref") {}
    virtual ~SelectKernelRef() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector