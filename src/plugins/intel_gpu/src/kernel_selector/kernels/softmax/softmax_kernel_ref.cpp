// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_kernel_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey SoftmaxKernelRef::GetSupportedKey() const {
    auto k = GetDefaultSupportedKey();

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

SoftmaxKernelRef::Parent::DispatchData SoftmaxKernelRef::SetDefault(const softmax_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    dispatchData.gws = GetSoftmaxDimGlobalSizes(params.dim, params.outputs[0]);

    assert(dispatchData.gws.size() == 3);

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority SoftmaxKernelRef::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

KernelsData SoftmaxKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData kds = GetCommonKernelsData(params, options);
    if (!kds.empty()) {
        const softmax_params& orgParams = static_cast<const softmax_params&>(params);
        bool is_dynamic = orgParams.outputs[0].is_dynamic();

        kds[0].update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
            const auto& prim_params = static_cast<const softmax_params&>(params);
            auto dispatchData = SetDefault(prim_params);
            OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
            kd.kernels[0].params.workGroups.global = dispatchData.gws;
            kd.kernels[0].params.workGroups.local = dispatchData.lws;
            kd.internalBufferSizes.clear();
            kd.internalBufferSizes.push_back(prim_params.inputs[0].PhysicalSizeInBytes());
            kd.internalBufferDataType = prim_params.inputs[0].GetDType();
        };

        if (is_dynamic) {
            auto& args = kds[0].kernels[0].params.arguments;
            args.clear();
            args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            args.push_back({ArgumentDescriptor::Types::INPUT, 0});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});

            kds[0].internalBufferSizes.clear();
            kds[0].internalBufferSizes.push_back(orgParams.inputs[0].PhysicalSizeInBytes());
            kds[0].internalBufferDataType = orgParams.inputs[0].GetDType();
        }
    }
    return kds;
}

JitConstants SoftmaxKernelRef::GetJitConstants(const softmax_params& params, DispatchData dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);
    if (!SimpleLayout(params.inputs[0].GetLayout())) {
        jit.AddConstant(MakeJitConstant("SOFTMAX_DIM_" + toString(params.dim), "1"));
    }
    return jit;
}
}  // namespace kernel_selector
