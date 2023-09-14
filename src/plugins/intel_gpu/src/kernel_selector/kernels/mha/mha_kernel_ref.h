// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mha_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct mha_params : public base_params {
    mha_params() : base_params(KernelType::MHA) {}
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// mha_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct mha_optional_params : optional_params {
    mha_optional_params() : optional_params(KernelType::MHA) {}
};

class MHAKernelRef : public KernelBaseOpenCL {
public:
    MHAKernelRef() : KernelBaseOpenCL("mha_ref") {}
    virtual ~MHAKernelRef() {}
    virtual JitConstants GetJitConstants(const mha_params& params) const;
    virtual CommonDispatchData SetDefault(const mha_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector
