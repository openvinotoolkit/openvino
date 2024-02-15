// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "beam_table_update_kernel_ref.hpp"

#include "kernel_selector_params.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

void BeamTableUpdateKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = dynamic_cast<const beam_table_update_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = false;
        ScalarDescriptor is_state_set;

        is_state_set.t = ScalarDescriptor::Types::UINT8;
        is_state_set.v.u8 = prim_params.is_state_set ? 1 : 0;
        kd.kernels[0].params.scalars.resize(1);
        kd.kernels[0].params.scalars[0] = is_state_set;
    };
}

KernelsData BeamTableUpdateKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }

    auto kernel_data = KernelData::Default<beam_table_update_params>(params);
    const auto& kernel_params = dynamic_cast<const beam_table_update_params&>(*kernel_data.params);
    const auto dispatch_data = SetDefault(kernel_params);
    const auto entry_point = GetEntryPoint(kernelName, kernel_params.layerID, params, options);
    const auto jit_constants = GetJitConstants(kernel_params);
    const auto jit = CreateJit(kernelName, jit_constants, entry_point);
    auto& kernel = kernel_data.kernels.front();

    GetUpdateDispatchDataFunc(kernel_data);

    FillCLKernelData(kernel,
                     dispatch_data,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     {},
                     false,
                     false,
                     static_cast<int>(kernel_params.inputs.size()),
                     GetFusedPrimitiveInputsCount(kernel_params),
                     static_cast<int>(kernel_params.outputs.size()),
                     kernel_params.outputs[0].is_dynamic());

    ScalarDescriptor is_state_set;
    is_state_set.t = ScalarDescriptor::Types::UINT8;
    is_state_set.v.u8 = 0;
    kernel.params.scalars.push_back(is_state_set);
    kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});

    return {kernel_data};
}

ParamsKey BeamTableUpdateKernelRef::GetSupportedKey() const {
    ParamsKey key;
    key.EnableInputDataType(Datatype::INT32);
    key.EnableOutputDataType(Datatype::INT32);
    key.EnableInputLayout(DataLayout::bfyx);
    key.EnableOutputLayout(DataLayout::bfyx);
    key.EnableTensorOffset();
    key.EnableTensorPitches();
    key.EnableBatching();
    key.EnableDynamicShapesSupport();
    return key;
}

bool BeamTableUpdateKernelRef::Validate(const Params& params, const optional_params& options) const {
    if (params.GetType() != KernelType::BEAM_TABLE_UPDATE || options.GetType() != KernelType::BEAM_TABLE_UPDATE) {
        return false;
    }

    const auto& kernel_params = dynamic_cast<const beam_table_update_params&>(params);
    if (kernel_params.inputs.size() != 2) {
        return false;
    }
    if (kernel_params.outputs.size() != 1) {
        return false;
    }

    return true;
}

JitConstants BeamTableUpdateKernelRef::GetJitConstants(const beam_table_update_params& kernel_params) const {
    return MakeBaseParamsJitConstants(kernel_params);
}

CommonDispatchData BeamTableUpdateKernelRef::SetDefault(const beam_table_update_params& kernel_params) {
    CommonDispatchData dispatch_data;

    auto output = kernel_params.outputs[0];
    if (!output.is_dynamic()) {
        dispatch_data.gws = {output.Batch().v, Align(output.LogicalSize() / output.Batch().v, 16), 1};
        dispatch_data.lws = {1, 16, 1};
    }

    return dispatch_data;
}

}  // namespace kernel_selector
