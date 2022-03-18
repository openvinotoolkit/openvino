// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "binary_convolution_kernel_ref.h"
#include <string>

namespace kernel_selector {

ParamsKey BinaryConvolutionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::BINARY);
    k.EnableInputWeightsType(WeightsType::BINARY);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::BINARY);
    k.EnableInputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_32fp);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDilation();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    return k;
}

BinaryConvolutionKernelBase::DispatchData BinaryConvolutionKernelRef::SetDefault(const binary_convolution_params& params,
                                                                                 int) const {
    DispatchData dispatchData = BinaryConvolutionKernelBase::SetDefault(params);

    const auto& out = params.outputs[0];

    auto b = out.Batch().v;
    auto f = out.Feature().v;
    auto y = out.Y().v;
    auto x = out.X().v;

    dispatchData.gws[0] = b;
    dispatchData.gws[1] = f;
    dispatchData.gws[2] = x * y;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}

KernelsPriority BinaryConvolutionKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants BinaryConvolutionKernelRef::GetJitConstants(const binary_convolution_params& params,
                                                         const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    int pad_physical_val = params.pad_value == -1.0f ? 0x00000000 : 0xFFFFFFFF;
    int leftovers_mask = (0xFFFFFFFF >> (32 - params.inputs[0].Feature().v % 32));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_NUM_PACKED", CeilDiv(params.inputs[0].Feature().v, 32)));
    jit.AddConstant(MakeJitConstant("FEATURE_PACK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("OFM_BLOCK_SIZE", 32));
    jit.AddConstant(MakeJitConstant("EXCLUDE_PAD", params.pad_value == 0.0f));
    jit.AddConstant(MakeJitConstant("PAD_VALUE", pad_physical_val));
    jit.AddConstant(MakeJitConstant("LEFTOVERS", params.inputs[0].Feature().v % 32 != 0));
    jit.AddConstant(MakeJitConstant("LEFTOVERS_MASK", leftovers_mask));

    return jit;
}

KernelsData BinaryConvolutionKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}

bool BinaryConvolutionKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (!BinaryConvolutionKernelBase::Validate(p, o) || !CovolutionBinaryCheckInput(p, o))
        return false;

    const auto& params = static_cast<const binary_convolution_params&>(p);

    if (!params.fused_ops.empty())
        return false;

    return true;
}

JitConstants BinaryConvolutionKernelRef::GetFusedPrimitivesJitConstants(const binary_convolution_params& params,
                                                                        const DispatchData& /*kd*/) const {
    JitConstants jit = {};

    auto input_dt = GetUnitType(params);
    FusedOpsConfiguration conf = {"", {"b", "f", "y", "x"}, "res", input_dt, 1 };
    jit.Merge(MakeFusedOpsJitConstants(params, {conf}));

    return jit;
}
}  // namespace kernel_selector
