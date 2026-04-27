// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_b_fs_yx_fsv16_imad.hpp"
#include "common_tools.h"

#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

static constexpr size_t simd = 16;
static constexpr size_t fsv = 16;
static constexpr size_t pref_work_groups = 16;

ParamsKey MVNKernel_b_fs_yx_fsv16_imad::GetSupportedKey() const {
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
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableBatching();
    // TODO Add support for across channels
    // k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNNormalizeVariance();
    k.EnableDynamicShapesSupport();

    return k;
}

DeviceFeaturesKey MVNKernel_b_fs_yx_fsv16_imad::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();
    k.requires_subgroup_reduce();

    return k;
}

bool MVNKernel_b_fs_yx_fsv16_imad::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        DO_NOT_USE_THIS_KERNEL(p.layerID);

    auto params = static_cast<const mvn_params&>(p);

    // TODO Add support for input padding via iterating over y (parallel or in kernel).
    // Skip padding check for dynamic tensors (padding not known at compile time).
    if (!params.has_dynamic_tensors()) {
        if (params.inputs[0].X().pad.Total() != 0 || params.inputs[0].Y().pad.Total() != 0 ||
            params.inputs[0].Z().pad.Total() != 0)
            DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
}

MVNKernelBase::DispatchData MVNKernel_b_fs_yx_fsv16_imad::SetDefault(const mvn_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;
    auto max_lws = std::min(max_wg, max_sgs * simd);

    if (params.has_dynamic_tensors()) {
        // Fixed LWS for shape-agnostic compilation (basic single-workgroup mode).
        // LWS is baked into reqd_work_group_size; GWS[1]/[2] updated at runtime.
        auto lws = std::max(max_lws / simd, (size_t)1) * simd;
        dispatchData.gws[0] = lws;
        dispatchData.gws[1] = 1;
        dispatchData.gws[2] = 1;
        dispatchData.lws[0] = lws;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    } else {
        auto items_num = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
        auto lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;
        dispatchData.gws[0] = lws;
        dispatchData.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
        dispatchData.gws[2] = params.outputs[0].Batch().v;
        dispatchData.lws[0] = lws;
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    dispatchData.itemsNum = 1;

    return dispatchData;
}

Datatype MVNKernel_b_fs_yx_fsv16_imad::GetAccumulatorType(const mvn_params& params) const {
    const auto& input_dt = params.inputs[0].GetDType();

    switch (input_dt) {
        case Datatype::F32:
        case Datatype::F16:
            return Datatype::F32;
        case Datatype::INT8:
        case Datatype::UINT8:
            return Datatype::INT32;
        default: return Datatype::F32;
    }
}

JitConstants MVNKernel_b_fs_yx_fsv16_imad::GetJitConstants(const mvn_params& params, DispatchData dispatchData) const {
    auto jits = Parent::GetJitConstants(params, dispatchData);

    auto activation_dt = GetActivationType(params);
    jits.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jits.Merge(MakeTypeJitConstants(activation_dt, "MEAN"));
    jits.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jits.AddConstant(MakeJitConstant("SIMD", simd));
    jits.AddConstant(MakeJitConstant("LWS", dispatchData.lws[0]));
    jits.AddConstant(MakeJitConstant("GWS", dispatchData.gws[0]));
    jits.AddConstant(MakeJitConstant("ITEM_GROUPS", dispatchData.itemsNum));
    auto input_layout = params.inputs[0].GetLayout();
    size_t input_slice_pitch = (input_layout == DataLayout::b_fs_yx_fsv32 ||
                                input_layout == DataLayout::b_fs_zyx_fsv32) ? 32 : 16;
    jits.AddConstant(MakeJitConstant("INPUT_SLICE_PITCH", input_slice_pitch));
    auto output_layout = params.outputs[0].GetLayout();
    size_t output_slice_pitch = (output_layout == DataLayout::b_fs_yx_fsv32 ||
                                 output_layout == DataLayout::b_fs_zyx_fsv32) ? 32 : 16;
    jits.AddConstant(MakeJitConstant("OUTPUT_SLICE_PITCH", output_slice_pitch));
    if (params.has_dynamic_tensors()) {
        // For dynamic shapes, ITEMS_NUM is passed as a scalar kernel argument
        // because inline accumulate functions can't access shape_info buffer.
        jits.AddConstant(MakeJitConstant("ITEMS_NUM", "items_num"));
    } else {
        // Define ITEMS_NUM via JIT so the batch-compilation #undef system
        // cleans it up between kernels sharing the same CL source file.
        jits.AddConstant(MakeJitConstant("ITEMS_NUM",
            params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v));
    }
    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order;

        if (params.inputs[0].GetDims().size() <= 4) {
            idx_order = {"b",
                         "(f + set_idx)",
                         "(output_spatial / OUTPUT_SIZE_X)",
                         "(output_spatial % OUTPUT_SIZE_X)"};
        } else if (params.inputs[0].GetDims().size() == 5) {
            idx_order = {"b",
                         "(f + set_idx)",
                         "(output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y))",
                         "((output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y)",
                         "(output_spatial % OUTPUT_SIZE_X)"};
        }

        auto conf = FusedOpsConfiguration("", idx_order, "normalized", activation_dt);
        if (params.has_dynamic_tensors()) {
            conf.SetBoundaryCheck(FusedOpsConfiguration::BoundaryCheck::ENABLED);
        }
        jits.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jits;
}

MVNKernel_b_fs_yx_fsv16_imad::MultiDispatchData MVNKernel_b_fs_yx_fsv16_imad::SetDefaultForMulti(
    const mvn_params& params) const {
    MultiDispatchData dispatchData;

    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;
    auto max_lws = std::min(max_wg, max_sgs * simd);

    size_t lws;
    if (params.has_dynamic_tensors()) {
        // Use hardware-max LWS for dynamic (data size unknown at compile time)
        lws = std::max(max_lws / simd, (size_t)1) * simd;
    } else {
        auto items_num = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
        lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;
    }

    // TODO Check if larger number of work-groups does not provide benefit
    size_t item_groups = pref_work_groups;
    dispatchData.item_groups = item_groups;

    size_t stage1_lws = lws;

    dispatchData.stage_1.gws[0] = stage1_lws * item_groups;
    dispatchData.stage_1.gws[1] = params.has_dynamic_tensors() ? 1 : CeilDiv(params.outputs[0].Feature().v, fsv);
    dispatchData.stage_1.gws[2] = params.has_dynamic_tensors() ? 1 : params.outputs[0].Batch().v;

    dispatchData.stage_1.lws[0] = stage1_lws;
    dispatchData.stage_1.lws[1] = 1;
    dispatchData.stage_1.lws[2] = 1;

    dispatchData.stage_1.itemsNum = item_groups;

    size_t stage2_lws = std::max(std::min(item_groups, max_lws) / simd, (size_t)1) * simd;

    dispatchData.stage_2.gws[0] = stage2_lws;
    dispatchData.stage_2.gws[1] = params.has_dynamic_tensors() ? 1 : CeilDiv(params.outputs[0].Feature().v, fsv);
    dispatchData.stage_2.gws[2] = params.has_dynamic_tensors() ? 1 : params.outputs[0].Batch().v;

    dispatchData.stage_2.lws[0] = stage2_lws;
    dispatchData.stage_2.lws[1] = 1;
    dispatchData.stage_2.lws[2] = 1;

    dispatchData.stage_2.itemsNum = item_groups;

    if (params.has_dynamic_tensors()) {
        dispatchData.stage_final.gws[0] = simd;  // Placeholder, updated at runtime
        dispatchData.stage_final.gws[1] = 1;
        dispatchData.stage_final.gws[2] = 1;
    } else {
        auto items_num = params.outputs[0].X().v * params.outputs[0].Y().v * params.outputs[0].Z().v;
        dispatchData.stage_final.gws[0] = std::max(items_num / simd, (size_t)1) * simd;
        dispatchData.stage_final.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
        dispatchData.stage_final.gws[2] = params.outputs[0].Batch().v;
    }

    dispatchData.stage_final.lws[0] = simd;
    dispatchData.stage_final.lws[1] = 1;
    dispatchData.stage_final.lws[2] = 1;

    dispatchData.stage_final.itemsNum = 1;

    return dispatchData;
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetMultiStageKernelsData(const mvn_params& params) const {
    if (!Validate(params))
        return {};

    constexpr size_t intermidiate_bytes = 4;
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    auto dispatchData = SetDefaultForMulti(orgParams);

    size_t kernels_num = params.mvnNormalizeVariance ? 5 : 3;
    KernelData kd = KernelData::Default<mvn_params>(params, kernels_num);

    auto finalKernelName = GetKernelName(orgParams);
    size_t entry_part_id = 0;
    {
        // Mean first stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
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
                         0,
                         0);
        kernel.params.arguments.clear();  // Clear original output argument
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kd.internalBuffers.push_back(params.outputs[0].Batch().v * Align(params.outputs[0].Feature().v, fsv) *
                                         dispatchData.item_groups * intermidiate_bytes);
    }
    {
        // Mean second stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
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
        kernel.params.arguments.clear();  // Clear original output argument
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kd.internalBuffers.push_back(params.outputs[0].Batch().v * Align(params.outputs[0].Feature().v, fsv) *
                                         intermidiate_bytes);
    }
    if (params.mvnNormalizeVariance) {
        // Variance first stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
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
                         0,
                         0);
        kernel.params.arguments.clear();  // Clear original output argument
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    }
    if (params.mvnNormalizeVariance) {
        // Variance second stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
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
        kernel.params.arguments.clear();  // Clear original output argument
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBuffers.push_back(params.outputs[0].Batch().v * Align(params.outputs[0].Feature().v, fsv) *
                                         intermidiate_bytes);
    }
    {  // Final
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_MEAN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[kernels_num - 1];
        FillCLKernelData(kernel,
                         dispatchData.stage_final,
                         params.engineInfo,
                         finalKernelName,
                         jit,
                         entry_point,
                         "",
                         false,
                         false,
                         1,
                         GetFusedPrimitiveInputsCount(params));
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        if (params.mvnNormalizeVariance) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        }
    }
    kd.internalBufferDataType = Datatype::F32;

    return {kd};
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetDynamicMultiStageKernelsData(const mvn_params& params) const {
    if (!Validate(params))
        return {};

    constexpr size_t intermediate_bytes = 4;
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);
    bool has_variance = params.mvnNormalizeVariance;
    size_t multi_stage_kernels = has_variance ? 5 : 3;
    size_t total_kernels = 1 + multi_stage_kernels;  // kernel[0]=basic + multi-stage

    KernelData kd = KernelData::Default<mvn_params>(params, total_kernels);
    auto finalKernelName = GetKernelName(orgParams);
    size_t entry_part_id = 0;

    // Helper to add items_num scalar argument to a kernel
    auto add_items_num_scalar = [](clKernelData& kernel) {
        ScalarDescriptor items_num;
        items_num.t = ScalarDescriptor::Types::UINT32;
        items_num.v.u32 = 1;  // placeholder; updated at runtime
        kernel.params.scalars.push_back(items_num);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SCALAR, 0});
    };

    // ---- Kernel[0]: Basic mode (single-kernel mean+var+normalize) ----
    {
        auto dispatchData = SetDefault(orgParams);
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel,
                         dispatchData,
                         params.engineInfo,
                         finalKernelName,
                         jit,
                         entry_point,
                         "",
                         false,
                         false,
                         1,
                         GetFusedPrimitiveInputsCount(params),
                         1,
                         orgParams.is_shape_agnostic);
        add_items_num_scalar(kernel);
    }

    // ---- Multi-stage kernels [1..N] ----
    auto dispatchData = SetDefaultForMulti(orgParams);

    // Kernel[1]: Mean first stage
    {
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[1];
        FillCLKernelData(kernel,
                         dispatchData.stage_1,
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
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        add_items_num_scalar(kernel);
    }

    // Kernel[2]: Mean second stage
    {
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[2];
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
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        add_items_num_scalar(kernel);
    }

    if (has_variance) {
        // Kernel[3]: Variance first stage
        {
            auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_1", 1));
            auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
            auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
            auto& kernel = kd.kernels[3];
            FillCLKernelData(kernel,
                             dispatchData.stage_1,
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
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            add_items_num_scalar(kernel);
        }

        // Kernel[4]: Variance second stage
        {
            auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_2", 1));
            auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id++);
            auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
            auto& kernel = kd.kernels[4];
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
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
            add_items_num_scalar(kernel);
        }
    }

    // Final kernel (last index)
    {
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_MEAN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, params, entry_part_id);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[total_kernels - 1];
        FillCLKernelData(kernel,
                         dispatchData.stage_final,
                         params.engineInfo,
                         finalKernelName,
                         jit,
                         entry_point,
                         "",
                         false,
                         false,
                         1,
                         GetFusedPrimitiveInputsCount(params),
                         1,
                         orgParams.is_shape_agnostic);
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        if (has_variance) {
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        }
        add_items_num_scalar(kernel);
    }

    // Internal buffers with placeholder sizes (resized at runtime)
    kd.internalBuffers.push_back(InternalBuffer(1));  // buf0: partial sums
    kd.internalBuffers.push_back(InternalBuffer(1));  // buf1: mean values
    if (has_variance) {
        kd.internalBuffers.push_back(InternalBuffer(1));  // buf2: variance values
    }
    kd.internalBufferDataType = Datatype::F32;

    GetUpdateDispatchDataFunc(kd);

    return {kd};
}

void MVNKernel_b_fs_yx_fsv16_imad::GetUpdateDispatchDataFunc(KernelData& kd) const {
    kd.update_dispatch_data_func = [](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const mvn_params&>(params);
        constexpr size_t local_fsv = 16;
        constexpr size_t local_simd = 16;
        constexpr size_t local_pref_work_groups = 16;
        constexpr size_t intermediate_bytes = 4;

        auto items_num = prim_params.outputs[0].X().v
                       * prim_params.outputs[0].Y().v
                       * prim_params.outputs[0].Z().v;
        auto batch = prim_params.outputs[0].Batch().v;
        auto feature = prim_params.outputs[0].Feature().v;

        // Update items_num scalar in all kernels that have it
        ScalarDescriptor items_num_scalar;
        items_num_scalar.t = ScalarDescriptor::Types::UINT32;
        items_num_scalar.v.u32 = static_cast<uint32_t>(items_num);
        for (size_t i = 0; i < kd.kernels.size(); ++i) {
            if (!kd.kernels[i].params.scalars.empty()) {
                kd.kernels[i].params.scalars[0] = items_num_scalar;
            }
        }

        if (kd.kernels.size() == 1) {
            // Basic mode only (single kernel)
            kd.kernels[0].params.workGroups.global[1] = CeilDiv(feature, local_fsv);
            kd.kernels[0].params.workGroups.global[2] = batch;
            kd.kernels[0].skip_execution = KernelData::SkipKernelExecution(prim_params);
        } else {
            // Dynamic multi-stage: kernel[0]=basic, kernel[1..]=multi-stage
            size_t num_kernels = kd.kernels.size();
            bool has_variance = num_kernels > 4;  // 6 = basic + 5 stages

            auto max_wg = prim_params.engineInfo.maxWorkGroupSize;
            auto slm_per_sg = local_fsv * 4;
            auto max_slm = prim_params.engineInfo.maxLocalMemSize;
            auto max_sgs = max_slm / slm_per_sg;
            auto max_lws = std::min(max_wg, max_sgs * local_simd);

            // Runtime decision: use multi-stage when spatial size is large enough
            bool enough_slm = max_lws / local_simd * local_simd * slm_per_sg <= max_slm;
            bool enough_lws = max_lws / local_simd >= 1;
            bool enough_items = items_num >= max_lws / local_simd * local_simd * local_pref_work_groups;
            bool use_multi_stage = enough_slm && enough_lws && enough_items;

            // Skip execution flags
            kd.kernels[0].skip_execution = use_multi_stage || KernelData::SkipKernelExecution(prim_params);
            for (size_t i = 1; i < num_kernels; ++i) {
                kd.kernels[i].skip_execution = !use_multi_stage || KernelData::SkipKernelExecution(prim_params);
            }

            // Always update basic mode GWS (in case it gets selected)
            kd.kernels[0].params.workGroups.global[1] = CeilDiv(feature, local_fsv);
            kd.kernels[0].params.workGroups.global[2] = batch;

            if (use_multi_stage) {
                auto lws = std::max(std::min(items_num, max_lws) / local_simd, (size_t)1) * local_simd;
                size_t item_groups = local_pref_work_groups;

                // Stage 1 (mean_1): kernel[1]
                kd.kernels[1].params.workGroups.global = {lws * item_groups, CeilDiv(feature, local_fsv), batch};
                kd.kernels[1].params.workGroups.local[0] = lws;

                // Stage 2 (mean_2): kernel[2]
                auto stage2_lws = std::max(std::min(item_groups, max_lws) / local_simd, (size_t)1) * local_simd;
                kd.kernels[2].params.workGroups.global = {stage2_lws, CeilDiv(feature, local_fsv), batch};
                kd.kernels[2].params.workGroups.local[0] = stage2_lws;

                if (has_variance) {
                    // var_1: kernel[3]
                    kd.kernels[3].params.workGroups.global = {lws * item_groups, CeilDiv(feature, local_fsv), batch};
                    kd.kernels[3].params.workGroups.local[0] = lws;

                    // var_2: kernel[4]
                    kd.kernels[4].params.workGroups.global = {stage2_lws, CeilDiv(feature, local_fsv), batch};
                    kd.kernels[4].params.workGroups.local[0] = stage2_lws;

                    // final: kernel[5]
                    auto final_gws0 = std::max(items_num / local_simd, (size_t)1) * local_simd;
                    kd.kernels[5].params.workGroups.global = {final_gws0, CeilDiv(feature, local_fsv), batch};
                } else {
                    // final: kernel[3]
                    auto final_gws0 = std::max(items_num / local_simd, (size_t)1) * local_simd;
                    kd.kernels[3].params.workGroups.global = {final_gws0, CeilDiv(feature, local_fsv), batch};
                }

                // Resize internal buffers if needed
                size_t buf0_size = batch * Align(feature, local_fsv) * item_groups * intermediate_bytes;
                size_t buf1_size = batch * Align(feature, local_fsv) * intermediate_bytes;

                if (kd.internalBuffers.size() >= 2 &&
                    (kd.internalBuffers[0].byte_count < buf0_size ||
                     kd.internalBuffers[1].byte_count < buf1_size)) {
                    kd.internalBuffers.clear();
                    kd.internalBuffers.push_back(buf0_size);
                    kd.internalBuffers.push_back(buf1_size);
                    if (has_variance) {
                        kd.internalBuffers.push_back(buf1_size);
                    }
                }
            }
        }
    };
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetKernelsData(const Params& params) const {
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    // For dynamic shapes, compile both basic and multi-stage kernels;
    // the runtime update_dispatch_data_func selects the optimal path.
    if (orgParams.has_dynamic_tensors()) {
        return GetDynamicMultiStageKernelsData(orgParams);
    }

    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto slm_per_sg = fsv * 4;
    auto max_lws = params.engineInfo.maxWorkGroupSize;
    auto items_num = orgParams.outputs[0].X().v * orgParams.outputs[0].Y().v * orgParams.outputs[0].Z().v;

    auto enough_slm = max_lws / simd * simd * slm_per_sg <= max_slm;
    auto enough_lws = max_lws / simd >= 1;
    auto enough_items = items_num >= max_lws / simd * simd * pref_work_groups;

    if (enough_slm && enough_lws && enough_items)
        return GetMultiStageKernelsData(orgParams);
    else
        return GetCommonKernelsData(params);
}

KernelsPriority MVNKernel_b_fs_yx_fsv16_imad::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
