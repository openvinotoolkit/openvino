// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sync_tensor_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SyncTensorKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableInputLayout(DataLayout::fb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::f);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableSyncTensorDim(SyncTensorDim::X);
    k.EnableSyncTensorDim(SyncTensorDim::Y);
    k.EnableSyncTensorDim(SyncTensorDim::Z);
    k.EnableSyncTensorDim(SyncTensorDim::FEATURE);
    k.EnableSyncTensorDim(SyncTensorDim::BATCH);
    k.EnableDifferentTypes();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableDynamicShapesSupport();

    return k;
}

SyncTensorKernelRef::Parent::DispatchData SyncTensorKernelRef::SetDefault(const sync_tensor_params& params) const {
    auto dispatchData = Parent::SetDefault(params);
    // dispatchData.gws = GetSyncTensorDimGlobalSizes(params.dim, params.outputs[0]);
    // assert(dispatchData.gws.size() == 3);
    // dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    return dispatchData;
}

KernelsPriority SyncTensorKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

void SyncTensorKernelRef::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const sync_tensor_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
    };
}

KernelsData SyncTensorKernelRef::GetKernelsData(const Params& params) const {
    KernelsData kds = GetCommonKernelsData(params);
    if (!kds.empty()) {
        const sync_tensor_params& orgParams = static_cast<const sync_tensor_params&>(params);
        bool is_dynamic = orgParams.outputs[0].is_dynamic();

        GetUpdateDispatchDataFunc(kds[0]);

        if (is_dynamic) {
            auto& args = kds[0].kernels[0].params.arguments;
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        }
    }
    return kds;
}

JitConstants SyncTensorKernelRef::GetJitConstants(const sync_tensor_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    if (!SimpleLayout(params.inputs[0].GetLayout())) {
        jit.AddConstant(MakeJitConstant("SYNCTENSOR_DIM_" + toString(params.dim), "1"));
    }
    return jit;
}
}  // namespace kernel_selector
