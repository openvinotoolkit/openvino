// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {

ParamsKey SDPAKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::F32);
    // beam table input
    k.EnableInputDataType(Datatype::INT32);

    // FIXME: support for compressed KV cache
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDynamicShapesSupport();

    return k;
}

JitConstants SDPAKernelRef::GetJitConstants(const sdpa_params& params) const {
    auto jit = SDPAKernelBase::GetJitConstants(params);

    auto acc_dt = params.inputs[0].GetDType();
    jit.Merge(MakeTypeJitConstants(acc_dt, "ACCUMULATOR"));

    TransposedDimensionAccessHelperJit dims_q(params.inputs[0], params.input0_order);
    jit.AddConstant(MakeJitConstant("HEAD_SIZE", dims_q.x()));

    return jit;
}

CommonDispatchData SDPAKernelRef::SetDefault(const sdpa_params& params) const {
    CommonDispatchData dispatchData;

    const auto& output = params.outputs[0];
    dispatchData.gws = { output.Batch().v * output.Feature().v, output.Y().v, output.X().v };
    dispatchData.lws = { 1, 1, output.X().v };

    return dispatchData;
}

KernelsData SDPAKernelRef::GetKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<sdpa_params>(params);
    const auto& prim_params = dynamic_cast<const sdpa_params&>(params);

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(prim_params);
    auto entry_point = GetEntryPoint(kernelName, prim_params.layerID, params);
    auto cldnn_jit = GetJitConstants(prim_params);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    GetUpdateDispatchDataFunc(kd);

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(prim_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(params), 1, prim_params.is_shape_agnostic);

    if (prim_params.indirect_axis != -1)
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(prim_params.inputs.size())});

    kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});

    kd.internalBufferSizes.clear();
    kd.internalBufferSizes.push_back(prim_params.inputs[0].ElementSize());
    kd.internalBufferDataType = prim_params.inputs[0].GetDType();

    return { kd };
}

void SDPAKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kernel_data) {
        const auto& prim_params = static_cast<const sdpa_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kernel_data.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kernel_data.kernels[0].params.workGroups.global = dispatchData.gws;
        kernel_data.kernels[0].params.workGroups.local = dispatchData.lws;
        kernel_data.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);

        auto& in_q = prim_params.inputs[0];
        auto& in_k = prim_params.inputs[1];
        TransposedDimensionAccessHelperBase dims_q(in_q, prim_params.input0_order);
        TransposedDimensionAccessHelperBase dims_k(in_k, prim_params.input1_order);

        auto elem_size = in_q.ElementSize();
        auto batch_size = in_q.LogicalSize() / dims_q.x_dim().v / dims_q.y_dim().v;
        kernel_data.internalBufferSizes.clear();
        kernel_data.internalBufferSizes.push_back(batch_size * dims_q.y_dim().v * dims_k.y_dim().v * elem_size);

        kernel_data.internalBufferDataType = in_q.GetDType();
    };
}

KernelsPriority SDPAKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_1;
    // return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
