// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

/*
    This kernel is basicaly a eltwise_kernel_vload8 but GetKernelsData is modfied
    to roundup features number to 32 when
*/

namespace kernel_selector {
class EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32 : public EltwiseKernelBase {
public:
    EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32() : EltwiseKernelBase("eltwise_mixed_byxf_and_fs_b_yx_fsv32") {}
    virtual ~EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32() {}

    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    KernelsPriority GetKernelsPriority(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
};
}  // namespace kernel_selector