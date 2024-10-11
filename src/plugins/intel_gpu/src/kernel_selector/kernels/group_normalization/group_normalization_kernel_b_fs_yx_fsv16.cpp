// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_kernel_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static constexpr size_t fsv = 16;
static constexpr size_t simd = fsv;

ParamsKey GroupNormalizationKernel_b_fs_yx_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

GroupNormalizationKernelBase::MultiDispatchData GroupNormalizationKernel_b_fs_yx_fsv16::SetDefault(const group_normalization_params &params) const {
    MultiDispatchData dispatchData;

    if (!params.has_dynamic_tensors()) {
        const auto& input = params.inputs[0];

        dispatchData.stage_1.gws[0] = input.X().v * input.Y().v;
        dispatchData.stage_1.gws[1] = CeilDiv(input.Feature().v, fsv) * input.Batch().v;
        dispatchData.stage_1.gws[2] = 1;

        dispatchData.stage_1.lws[0] = input.X().v * input.Y().v;
        dispatchData.stage_1.lws[1] = 1;
        dispatchData.stage_1.lws[2] = 1;

        size_t divisor = 2;
        while (dispatchData.stage_1.lws[0] > (params.engineInfo.maxWorkGroupSize / fsv)) {
            if (dispatchData.stage_1.gws[0] % divisor == 0) {
                dispatchData.stage_1.lws[0] = dispatchData.stage_1.gws[0] / divisor;
            }
            divisor += 1;
        }
        dispatchData.stage_1.lws[0] *= fsv;
        dispatchData.stage_1.gws[0] = dispatchData.stage_1.lws[0];

        dispatchData.stage_2.gws[0] = input.Feature().v;
        dispatchData.stage_2.gws[1] = input.Batch().v;
        dispatchData.stage_2.gws[2] = 1;

        dispatchData.stage_2.lws[0] = input.Feature().v / params.num_groups;
        dispatchData.stage_2.lws[1] = 1;
        dispatchData.stage_2.lws[2] = 1;

        divisor = 2;
        while (dispatchData.stage_2.lws[0] > params.engineInfo.maxWorkGroupSize) {
            if ((input.Feature().v / params.num_groups) % divisor == 0) {
                dispatchData.stage_2.lws[0] = (input.Feature().v / params.num_groups) / divisor;
            }
            divisor += 1;
        }

        dispatchData.stage_final.gws[0] = input.X().v * input.Y().v;
        dispatchData.stage_final.gws[1] = CeilDiv(input.Feature().v, fsv) * input.Batch().v;
        dispatchData.stage_final.gws[2] = 1;

        dispatchData.stage_final.lws[0] = input.X().v * input.Y().v;
        dispatchData.stage_final.lws[1] = CeilDiv(input.Feature().v, fsv) * input.Batch().v;
        dispatchData.stage_final.lws[2] = 1;

        divisor = 1;
        while (dispatchData.stage_final.lws[0] > (params.engineInfo.maxWorkGroupSize / fsv)) {
            if (dispatchData.stage_final.gws[0] % divisor == 0) {
                dispatchData.stage_final.lws[0] = dispatchData.stage_final.gws[0] / divisor;
            }
            divisor += 1;
        }
        dispatchData.stage_final.lws[0] *= fsv;
        dispatchData.stage_final.gws[0] *= fsv;

        divisor = 2;
        while ((dispatchData.stage_final.lws[0] * dispatchData.stage_final.lws[1]) > params.engineInfo.maxWorkGroupSize) {
            if (dispatchData.stage_final.gws[1] % divisor == 0) {
                dispatchData.stage_final.lws[1] = dispatchData.stage_final.gws[1] / divisor;
            }
            divisor += 1;
        }
    }

    return dispatchData;
}

JitConstants GroupNormalizationKernel_b_fs_yx_fsv16::GetJitConstants(const group_normalization_params &params,
                                                              GroupNormalizationKernelBase::DispatchData dispatchData) const {
    auto jit = GroupNormalizationKernelBase::GetJitConstants(params);

    jit.AddConstants({
        MakeJitConstant("SIMD", simd),
        MakeJitConstant("FSV", fsv),
    });

    if (params.has_dynamic_tensors()) {
        jit.AddConstants({
            MakeJitConstant("GWS0", "get_global_size(0)"),
            MakeJitConstant("LWS0", "get_local_size(0)"),
            MakeJitConstant("SLM_SIZE", params.engineInfo.maxWorkGroupSize),
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("GWS0", dispatchData.gws[0]),
            MakeJitConstant("LWS0", dispatchData.lws[0]),
            MakeJitConstant("SLM_SIZE", dispatchData.lws[0]),
        });
    }
    auto activation_dt = GetActivationType(params);
    jit.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;
        if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = { "(b)", "(f)", "(y)", "(x)" };
        } else {
            OPENVINO_THROW("group_normalization_b_fs_yx_fsv16 doesn't support 5D or higher dims.");
        }
        auto conf = FusedOpsConfiguration("", idx_order, "normalized", activation_dt, 1);
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

void GroupNormalizationKernel_b_fs_yx_fsv16::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const group_normalization_params&>(params);
        auto dispatchData = SetDefault(prim_params);

        kd.kernels[0].params.workGroups.global = dispatchData.stage_1.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.stage_1.lws;
        kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params, 0);

        kd.kernels[1].params.workGroups.global = dispatchData.stage_2.gws;
        kd.kernels[1].params.workGroups.local = dispatchData.stage_2.lws;
        kd.kernels[1].skip_execution = KernelData::SkipKernelExecution(prim_params, 1);

        kd.kernels[2].params.workGroups.global = dispatchData.stage_1.gws;
        kd.kernels[2].params.workGroups.local = dispatchData.stage_1.lws;
        kd.kernels[2].skip_execution = KernelData::SkipKernelExecution(prim_params, 2);

        kd.kernels[3].params.workGroups.global = dispatchData.stage_2.gws;
        kd.kernels[3].params.workGroups.local = dispatchData.stage_2.lws;
        kd.kernels[3].skip_execution = KernelData::SkipKernelExecution(prim_params, 3);

        kd.kernels[4].params.workGroups.global = dispatchData.stage_final.gws;
        kd.kernels[4].params.workGroups.local = dispatchData.stage_final.lws;
        kd.kernels[4].skip_execution = KernelData::SkipKernelExecution(prim_params, 4);

        kd.internalBufferSizes.clear();
        kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * Align(prim_params.outputs[0].Feature().v, fsv) * 4);
        kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * Align(prim_params.outputs[0].Feature().v, fsv) * 4);
    };
}

bool GroupNormalizationKernel_b_fs_yx_fsv16::Validate(const Params& params) const {
    if (!Parent::Validate(params))
        return false;

    const group_normalization_params& prim_params = static_cast<const group_normalization_params&>(params);

    if (prim_params.has_dynamic_tensors())
        return true;

    // no support for spatial paddings
    if (prim_params.inputs[0].X().pad.Total() > 0 || prim_params.inputs[0].Y().pad.Total() > 0) {
        return false;
    }

    // feature paddings should be multiples of fsv.
    if (prim_params.inputs[0].Feature().pad.before % fsv != 0) {
        return false;
    }

    return true;
}

KernelsData GroupNormalizationKernel_b_fs_yx_fsv16::GetKernelsData(const Params &params) const {
    assert(params.GetType() == KernelType::GROUP_NORMALIZATION);

    if (!Validate(params))
        return {};

    const group_normalization_params& prim_params = static_cast<const group_normalization_params&>(params);

    MultiDispatchData dispatchData = SetDefault(prim_params);

    KernelData kd = KernelData::Default<group_normalization_params>(params, 5);
    kd.internalBufferDataType = GetAccumulatorType(prim_params);
    GetUpdateDispatchDataFunc(kd);

    auto finalKernelName = GetKernelName(prim_params);
    size_t entry_part_id = 0;

    {
        // Mean first stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("GROUP_NORM_KERNEL_FEATURE_MEAN", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel,
                        dispatchData.stage_1,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        1,
                        0,
                        0,
                        prim_params.is_shape_agnostic);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        if (!prim_params.has_dynamic_tensors()) {
            kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * Align(prim_params.outputs[0].Feature().v, fsv) * 4);
        }
    }
    {
        // Mean second stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("GROUP_NORM_KERNEL_GROUP_MEAN", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[1];
        FillCLKernelData(kernel,
                        dispatchData.stage_2,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        0,
                        0);
        kernel.params.arguments.clear();
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    }
    {
        // Variance first stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("GROUP_NORM_KERNEL_FEATURE_VAR", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[2];
        FillCLKernelData(kernel,
                        dispatchData.stage_1,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        1,
                        0,
                        0,
                        prim_params.is_shape_agnostic);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        if (!prim_params.has_dynamic_tensors()) {
            kd.internalBufferSizes.push_back(prim_params.outputs[0].Batch().v * Align(prim_params.outputs[0].Feature().v, fsv) * 4);
        }
    }
    {
        // Variance second stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("GROUP_NORM_KERNEL_GROUP_VAR", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[3];
        FillCLKernelData(kernel,
                        dispatchData.stage_2,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        0,
                        0);
        kernel.params.arguments.clear();
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    }
    {
        // final stage
        auto cldnn_jit = GetJitConstants(prim_params, dispatchData.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("GROUP_NORM_KERNEL_FINAL", 1));
        auto entry_point = GetEntryPoint(finalKernelName, prim_params.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[4];
        FillCLKernelData(kernel,
                        dispatchData.stage_final,
                        params.engineInfo,
                        finalKernelName,
                        jit,
                        entry_point,
                        "",
                        false,
                        false,
                        3,
                        GetFusedPrimitiveInputsCount(params),
                        1,
                        prim_params.is_shape_agnostic);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
    }

    return {kd};
}

KernelsPriority GroupNormalizationKernel_b_fs_yx_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
} // namespace kernel_selector
