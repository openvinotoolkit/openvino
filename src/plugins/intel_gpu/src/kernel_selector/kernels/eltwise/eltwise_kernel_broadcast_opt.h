// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "eltwise_kernel_base.h"

namespace kernel_selector {
// Vectorized broadcast elementwise multiply (f16, planar, dynamic-shape capable).
// Targets the broadcast multiplies that otherwise fall back to generic_eltwise_ref:
// out[i] = a[i] * b[i % period], where the broadcast operand's non-unit dims form an
// innermost-contiguous suffix (AdaLayerNorm/RMSNorm/SwiGLU/scalar in FLUX.2).
class EltwiseKernelBroadcastOpt : public EltwiseKernelBase {
public:
    EltwiseKernelBroadcastOpt() : EltwiseKernelBase("eltwise_broadcast_opt") {}
    virtual ~EltwiseKernelBroadcastOpt() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& p) const override;
    JitConstants GetJitConstants(const eltwise_params& params) const override;
    DispatchData SetDefault(const eltwise_params& params) const override;
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};
}  // namespace kernel_selector
