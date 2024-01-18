// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_opt.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey BroadcastKernelOpt::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    k.EnableDynamicShapesSupport();

    return k;
}

KernelsData BroadcastKernelOpt::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    assert(params.GetType() == KernelType::BROADCAST);

    const auto& prim_params = static_cast<const broadcast_params&>(params);

    auto dispatchData = SetDefault(prim_params);
    KernelData k_data = KernelData::Default<broadcast_params>(params);
    GetUpdateDispatchDataFunc(k_data);

    auto cldnn_jit = GetJitConstants(prim_params);
    cldnn_jit.AddConstant(MakeJitConstant("INPUT0_BLOCK_ND", GetInputBlockND(prim_params)));
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = k_data.kernels[0];
    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     0,
                     1,
                     prim_params.inputs[0].is_dynamic() || prim_params.outputs[0].is_dynamic());
    return {k_data};
}

KernelsPriority BroadcastKernelOpt::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants BroadcastKernelOpt::GetJitConstants(const broadcast_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("BROADCAST_ORDER", params.input_order)});
    jit.AddConstants({MakeJitConstant("VEC_SIZE", vec_size)});
    jit.AddConstants({MakeJitConstant("Y_BLOCK_SIZE", y_block_size)});
    jit.AddConstants({MakeJitConstant("USE_VEC", use_vec)});

    return jit;
}

BroadcastKernelBase::DispatchData BroadcastKernelOpt::SetDefault(const broadcast_params& params) const {
    const auto& output = params.outputs[0];

    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                       Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    dispatchData.gws = { output.X().v / vec_size, output.Y().v / y_block_size,  output.Z().v * output.W().v * output.Feature().v * output.Batch().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

bool BroadcastKernelOpt::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::BROADCAST ||
        o.GetType() != KernelType::BROADCAST) {
        return false;
    }

    const broadcast_params& params = static_cast<const broadcast_params&>(p);
    if (params.outputs[0].GetDims().size() != 5)
        return false;

    if (params.outputs[0].Y().v % 4 != 0)
        return false;

    if (!(params.outputs[0].Batch().v == params.inputs[0].Batch().v
        && params.outputs[0].Feature().v == params.inputs[0].Feature().v
        && params.outputs[0].Z().v == params.inputs[0].Z().v
        && params.outputs[0].Y().v != params.inputs[0].Y().v
        && params.outputs[0].X().v == params.inputs[0].X().v)) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
