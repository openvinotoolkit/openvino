// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "permute_kernel_bf_swap.h"

#include "common_tools.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
constexpr size_t kVectorBytes = 16;
}

ParamsKey PermuteKernel_bf_swap::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::UINT2);
    key.EnableOutputDataType(Datatype::UINT2);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    return key;
}

JitConstants PermuteKernel_bf_swap::GetJitConstants(const permute_params& params,
                                                     const CommonDispatchData& dispatch_data) const {
    auto jit = Parent::GetJitConstants(params, dispatch_data);
    jit.AddConstant(MakeJitConstant("VECTOR_BYTES", kVectorBytes));
    jit.AddConstant(MakeJitConstant("PLANE_BLOCKS", params.inputs[0].Y().v * params.inputs[0].X().v / 4 /
                                                        kVectorBytes));
    return jit;
}

CommonDispatchData PermuteKernel_bf_swap::SetDefault(const permute_params& params) const {
    CommonDispatchData dispatch_data;
    const auto& input = params.inputs[0];
    const size_t plane_blocks = input.Y().v * input.X().v / 4 / kVectorBytes;
    dispatch_data.gws = {plane_blocks, input.Feature().v, input.Batch().v};
    dispatch_data.lws = GetOptimalLocalWorkGroupSizes(dispatch_data.gws,
                                                       params.engineInfo,
                                                       input.GetLayout(),
                                                       params.outputs[0].GetLayout());
    return dispatch_data;
}

bool PermuteKernel_bf_swap::Validate(const Params& p) const {
    if (!Parent::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const permute_params&>(p);
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];
    const size_t plane_elements = input.Y().v * input.X().v;

    if (params.order != std::vector<uint16_t>{1, 0, 2, 3} ||
        input.GetDType() != Datatype::UINT2 || output.GetDType() != Datatype::UINT2 ||
        input.GetLayout() != DataLayout::bfyx || output.GetLayout() != DataLayout::bfyx ||
        params.has_dynamic_tensors() || !params.fused_ops.empty() ||
        input.PitchesDifferFromLogicalDims() || output.PitchesDifferFromLogicalDims() ||
        input.GetFirstElementOffset() != 0 || output.GetFirstElementOffset() != 0 ||
        plane_elements % (4 * kVectorBytes) != 0) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

KernelsPriority PermuteKernel_bf_swap::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
}

}  // namespace kernel_selector
