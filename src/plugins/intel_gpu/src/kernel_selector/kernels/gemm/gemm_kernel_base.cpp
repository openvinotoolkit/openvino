// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gemm_kernel_base.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
JitConstants GemmKernelBase::GetJitConstants(const gemm_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("ALPHA", params.alpha),
        MakeJitConstant("BETA", params.beta),
        MakeJitConstant("TRANSPOSE_INPUT0", params.transpose_input0),
        MakeJitConstant("TRANSPOSE_INPUT1", params.transpose_input1),
        MakeJitConstant("QUANTIZATION_TERM", params.quantization != QuantizationType::NONE),
    });

    return jit;
}

GemmKernelBase::DispatchData GemmKernelBase::SetDefault(const gemm_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;

    if (!output.is_dynamic()) {
        auto total_batches = output.LogicalSize() / (output.X().v * output.Y().v);
        dispatchData.gws = { output.X().v, output.Y().v, total_batches };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }

    return dispatchData;
}

void GemmKernelBase::SetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const gemm_params&>(params);
            auto dispatchData = SetDefault(prim_params);
            OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
            kd.kernels[0].params.workGroups.global = dispatchData.gws;
            kd.kernels[0].params.workGroups.local = dispatchData.lws;
            kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData GemmKernelBase::GetCommonKernelsData(const Params& params,
                                                 const optional_params& options) const {
    if (!Validate(params, options)) {
        return KernelsData();
    }

    const auto& prim_params = static_cast<const gemm_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<gemm_params>(params);
    SetUpdateDispatchDataFunc(k_data);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     (uint32_t)prim_params.inputs.size(),
                     GetFusedPrimitiveInputsCount(params),
                     1,
                     prim_params.has_dynamic_tensors());

    return {k_data};
}

JitConstants GemmKernelBase::GetFusedPrimitivesJitConstants(const gemm_params&, const DispatchData&) const {
    return {};
}

bool GemmKernelBase::Validate(const Params& p, const optional_params&) const {
    const gemm_params& params = static_cast<const gemm_params&>(p);

    if (params.GetType() != KernelType::GEMM) {
        return false;
    }

    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    return true;
}

Datatype GemmKernelBase::GetActivationType(const gemm_params& params) const {
    if (params.quantization != QuantizationType::NONE)
        return Datatype::F32;

    return GetUnitType(params);
}

}  // namespace kernel_selector
