// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "select_kernel_base.h"

namespace kernel_selector {
class SelectKernelRef : public SelectKernelBase {
public:
    SelectKernelRef() : SelectKernelBase("select_gpu_ref") {}
    virtual ~SelectKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { FusedOpType::REORDER };
    }


protected:
    bool Validate(const Params& p) const override;
};
}  // namespace kernel_selector
