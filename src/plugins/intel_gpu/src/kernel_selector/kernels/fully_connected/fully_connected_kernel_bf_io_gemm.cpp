// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bf_io_gemm.h"
#include <vector>

namespace kernel_selector {

ParamsKey FullyConnected_bf_io_GEMM::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    // bfyx -> bf layout transformation works incorrectly when tensor has paddings, so offset support is disabled for now.
    // k.EnableTensorOffset();
    k.EnableTensorPitches();
    return k;
}

FullyConnected_bf_io_GEMM::DispatchData FullyConnected_bf_io_GEMM::SetDefault(const fully_connected_params& params,
                                                                              int autoTuneIndex,
                                                                              int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params, autoTuneIndex);

    const uint32_t localWorkSizeX = 64;
    const uint32_t globalWorkSizeX = localWorkSizeX;

    dispatchData.gws = { globalWorkSizeX, params.outputs[0].Feature().v, 1 };
    dispatchData.lws = { localWorkSizeX, 1, 1 };

    return dispatchData;
}

KernelsPriority FullyConnected_bf_io_GEMM::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants FullyConnected_bf_io_GEMM::GetJitConstants(const fully_connected_params& params,
                                                        const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (params.inputs[0].GetDType() == Datatype::F16) {
        jit.AddConstant(MakeJitConstant("__fc_f16", ""));
    } else {
        jit.AddConstant(MakeJitConstant("__fc_f32", ""));
    }

    const uint32_t localWorkSizeX = 64;
    const uint32_t globalWorkSizeX = localWorkSizeX;
    const uint32_t vecSize = 4;
    size_t matrixLineSize = params.inputs[0].Batch().pitch;

    jit.AddConstants({
        MakeJitConstant("LAST_INPUT_SIZE_REMAINDER", matrixLineSize % (globalWorkSizeX * vecSize)),
        MakeJitConstant("LAST_INPUT_SIZE_DIV_4", matrixLineSize % vecSize),
    });

    return jit;
}

bool FullyConnected_bf_io_GEMM::Validate(const Params& p) const {
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

KernelsData FullyConnected_bf_io_GEMM::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::bf,
                                                    WeightsLayout::oiyx,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}
}  // namespace kernel_selector
