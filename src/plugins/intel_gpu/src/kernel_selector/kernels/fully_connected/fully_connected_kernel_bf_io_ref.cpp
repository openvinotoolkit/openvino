// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "fully_connected_kernel_bf_io_ref.h"

namespace kernel_selector {
ParamsKey FullyConnected_bf_io_ref::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

JitConstants FullyConnected_bf_io_ref::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetUnitType(params);
        FusedOpsConfiguration conf = { "", { "batch_id", "outXIdx", "0", "0" }, "result", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

bool FullyConnected_bf_io_ref::Validate(const Params& p) const {
    if (!FullyConnectedKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const fully_connected_params&>(p);

    if (!params.bias.empty()) {
        if (params.inputs[0].GetDType() != params.bias[0].GetDType()) {
            return false;
        }
    }

    return true;
}

KernelsData FullyConnected_bf_io_ref::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::bf,
                                                    WeightsLayout::io,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_bf_io_ref::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
