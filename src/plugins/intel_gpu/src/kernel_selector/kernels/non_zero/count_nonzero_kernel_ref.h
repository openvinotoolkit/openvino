// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// count_nonzero_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct count_nonzero_params : public base_params {
    count_nonzero_params() : base_params(KernelType::COUNT_NONZERO) {}
    int32_t ov_input_rank = -1;
};

class CountNonzeroKernelRef : public KernelBaseOpenCL {
public:
    CountNonzeroKernelRef() : KernelBaseOpenCL("count_nonzero_ref") {}
    virtual ~CountNonzeroKernelRef() {}

    struct DispatchData : public CommonDispatchData {
        size_t dataSize;
        DispatchData() : dataSize(1) {}
    };

    virtual DispatchData SetDefault(const count_nonzero_params& params) const;
    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    DeviceFeaturesKey get_required_device_features_key(const Params& params) const override;

protected:
    bool Validate(const Params& pp) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
