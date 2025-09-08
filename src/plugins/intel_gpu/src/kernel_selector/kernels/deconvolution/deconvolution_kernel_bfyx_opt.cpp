// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "deconvolution_kernel_bfyx_opt.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

ParamsKey DeconvolutionKernel_bfyx_opt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableDifferentTypes();
    return k;
}

CommonDispatchData DeconvolutionKernel_bfyx_opt::SetDefault(const deconvolution_params& params) const {
    DispatchData dispatchData;

    auto wg_size = 16;

    if (params.inputs[0].X().v == 1 && params.outputs[0].X().v == 1 && params.filterSize.x == 1) {
        dispatchData.gws[0] = Align(params.outputs[0].Y().v, wg_size * params.stride.y);
        dispatchData.gws[1] = params.outputs[0].X().v;
        dispatchData.gws[2] = params.outputs[0].Batch().v * params.outputs[0].Feature().v;
    } else {
        dispatchData.gws[0] = Align(params.outputs[0].X().v, wg_size * params.stride.x);
        dispatchData.gws[1] = params.outputs[0].Y().v;
        dispatchData.gws[2] = params.outputs[0].Batch().v * params.outputs[0].Feature().v;
    }

    dispatchData.lws[0] = wg_size;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;


    return dispatchData;
}

KernelsPriority DeconvolutionKernel_bfyx_opt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants DeconvolutionKernel_bfyx_opt::GetJitConstants(const deconvolution_params& params) const {
    auto jit = Parent::GetJitConstants(params);

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        FusedOpsConfiguration conf = {
            "",
            {"batch_offset", "ofm_offset", "id_y", "id_x"},
            "result",
            fused_dt,
            1,
            LoadType::LT_UNALIGNED,
            BoundaryCheck::DISABLED };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    if (params.inputs[0].X().v == 1 && params.outputs[0].X().v == 1 && params.filterSize.x == 1) {
        jit.AddConstant(MakeJitConstant("Y_AXIS_1D_FILTER", 1));
    }

    return jit;
}

}  // namespace kernel_selector
