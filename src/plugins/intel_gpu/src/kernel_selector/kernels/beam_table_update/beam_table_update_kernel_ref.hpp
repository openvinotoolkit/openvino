// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct beam_table_update_params : base_params {
    beam_table_update_params() : base_params(KernelType::BEAM_TABLE_UPDATE) {}
    bool is_state_set = false;
};

struct beam_table_update_optional_params : optional_params {
    beam_table_update_optional_params() : optional_params(KernelType::BEAM_TABLE_UPDATE) {}
};


class BeamTableUpdateKernelRef : public KernelBaseOpenCL {
public:
    BeamTableUpdateKernelRef() : KernelBaseOpenCL{"beam_table_update_ref"} {}
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params, const optional_params& options) const override;
    JitConstants GetJitConstants(const beam_table_update_params& kernel_params) const;
    static CommonDispatchData SetDefault(const beam_table_update_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
