// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvn_kernel_bs_fs_yx_bsv32.hpp"
#include "common_tools.h"

#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

static constexpr size_t simd = 16;
static constexpr size_t fsv = 16;
static constexpr size_t pref_work_groups = 16;

ParamsKey MVNKernel_bs_fs_yx_bsv32::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);

    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableBatching();

    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNNormalizeVariance();

    return k;
}

DeviceFeaturesKey MVNKernel_bs_fs_yx_bsv32::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();
    k.requires_subgroup_reduce();

    return k;
}

bool MVNKernel_bs_fs_yx_bsv32::Validate(const Params& p) const {
    if (!Parent::Validate(p))
        return false;

    auto params = static_cast<const mvn_params&>(p);

    // TODO Add support for input padding via iterating over y (parallel or in kernel).
    if (params.inputs[0].X().pad.Total() != 0 || params.inputs[0].Y().pad.Total() != 0)
        return false;

    return true;
}

Datatype MVNKernel_bs_fs_yx_bsv32::GetAccumulatorType(const mvn_params& params) const {
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

JitConstants MVNKernel_bs_fs_yx_bsv32::GetJitConstants(const mvn_params& params, DispatchData dispatchData) const {
    auto jits = Parent::GetJitConstants(params, dispatchData);

    auto activation_dt = GetActivationType(params);
    jits.Merge(MakeTypeJitConstants(activation_dt, "ACTIVATION"));
    jits.Merge(MakeTypeJitConstants(Datatype::F32, "MEAN"));
    jits.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jits.AddConstant(MakeJitConstant("SIMD", simd));
    jits.AddConstant(MakeJitConstant("LWS", dispatchData.lws[0]));
    jits.AddConstant(MakeJitConstant("GWS", dispatchData.gws[0]));
    jits.AddConstant(MakeJitConstant("ITEM_GROUPS", dispatchData.itemsNum));
    const auto input_layout = params.inputs[0].GetLayout();

    if (input_layout == DataLayout::bs_fs_yx_bsv32_fsv32) {
        jits.AddConstant(MakeJitConstant("INPUT_SLICE_PITCH", (size_t)(32 * 32)));
    } else { // DataLayout::bs_fs_yx_bsv32_fsv16
        jits.AddConstant(MakeJitConstant("INPUT_SLICE_PITCH", (size_t)(32 * 16)));
    }

    if (!params.fused_ops.empty()) {
        std::vector<std::string> idx_order = {"b", "(f + fi)", "(y)", "(x)"};
        auto conf = FusedOpsConfiguration("", idx_order, "normalized", activation_dt);
        jits.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    return jits;
}

std::vector<size_t> MVNKernel_bs_fs_yx_bsv32::GetFinalKernelLws(const std::vector<size_t>& gws, uint64_t max_wg) const {
    std::vector<size_t> lws(3);
    lws[0] = 1;
    lws[1] = gws[1];
    lws[2] = gws[2];

    // gws[1] is CeilDiv(feature, simd)
    while (lws[1] > 16 || gws[1] % lws[1] != 0) {
        lws[1] -= 1;
    }

    // gws[2] is Align(batch, simd)
    while (lws[1] * lws[2] > max_wg && lws[2] > 16) {
        lws[2] -= simd;
    }

    return lws;
}

MVNKernel_bs_fs_yx_bsv32::MultiDispatchData MVNKernel_bs_fs_yx_bsv32::SetDefaultForMulti(const mvn_params& params,
                                                                                                    bool has_enough_data) const {
    MultiDispatchData dispatchData;

    auto items_num = params.outputs[0].X().v * params.outputs[0].Y().v;
    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;

    auto max_lws = std::min(max_wg, max_sgs * simd);
    auto lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;

    // TODO Check if larger number of work-groups does not provide benefit
    size_t item_groups = pref_work_groups;
    dispatchData.item_groups = item_groups;

    size_t stage1_lws = lws;

    if (has_enough_data) {
        dispatchData.stage_1.gws[0] = stage1_lws * item_groups;
        dispatchData.stage_1.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
        dispatchData.stage_1.gws[2] = params.outputs[0].Batch().v;

        dispatchData.stage_1.lws[0] = stage1_lws;
        dispatchData.stage_1.lws[1] = 1;
        dispatchData.stage_1.lws[2] = 1;

        dispatchData.stage_1.itemsNum = item_groups;

        size_t stage2_lws = std::max(std::min(item_groups, max_lws) / simd, (size_t)1) * simd;

        dispatchData.stage_2.gws[0] = stage2_lws;
        dispatchData.stage_2.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
        dispatchData.stage_2.gws[2] = params.outputs[0].Batch().v;

        dispatchData.stage_2.lws[0] = stage2_lws;
        dispatchData.stage_2.lws[1] = 1;
        dispatchData.stage_2.lws[2] = 1;

        dispatchData.stage_2.itemsNum = item_groups;
    } else {
        dispatchData.stage_1.gws[0] = lws;
        dispatchData.stage_1.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
        dispatchData.stage_1.gws[2] = params.outputs[0].Batch().v;

        dispatchData.stage_1.lws[0] = lws;
        dispatchData.stage_1.lws[1] = 1;
        dispatchData.stage_1.lws[2] = 1;

        dispatchData.stage_1.itemsNum = 1;
    }

    dispatchData.stage_final.gws[0] = items_num;
    dispatchData.stage_final.gws[1] = CeilDiv(params.outputs[0].Feature().v, fsv);
    dispatchData.stage_final.gws[2] = Align(params.outputs[0].Batch().v, simd);

    dispatchData.stage_final.lws = GetFinalKernelLws(dispatchData.stage_final.gws, max_wg);
    dispatchData.stage_final.itemsNum = 1;

    return dispatchData;
}

KernelsData MVNKernel_bs_fs_yx_bsv32::GetMultiStageKernelsData(const mvn_params& params,
                                                                        bool has_enough_data) const {
    if (!Validate(params))
        return {};

    constexpr size_t intermediate_bytes = 4;
    auto dispatchData = SetDefaultForMulti(params, has_enough_data);

    KernelData kd;
    size_t entry_part_id = 0;

    if (has_enough_data) {
        size_t kernels_num = params.mvnNormalizeVariance ? 5 : 3;
        kd = KernelData::Default<mvn_params>(params, kernels_num);

        auto finalKernelName = GetKernelName(params);
        {
            // Mean first stage
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_1);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_1", 1));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
                                            dispatchData.item_groups * intermediate_bytes);
        }
        {
            // Mean second stage
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_2);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_2", 1));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
                                            intermediate_bytes);
        }
        if (params.mvnNormalizeVariance) {
            // Variance first stage
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_1);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_1", 1));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_2);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_2", 1));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
                                            intermediate_bytes);
        }
        {  // Final
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_final);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN_BSV32", 1));
            cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
    } else { // not enough data
        kd = KernelData::Default<mvn_params>(params, 2);
        auto finalKernelName = GetKernelName(params);
        {
            // Mean and Variance stage
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_1);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_VAR_BSV32", 1));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
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
                                            intermediate_bytes);
            if (params.mvnNormalizeVariance) {
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
                kd.internalBuffers.push_back(params.outputs[0].Batch().v * Align(params.outputs[0].Feature().v, fsv) *
                                            intermediate_bytes);
            }
        }
        {  // Final
            auto cldnn_jit = GetJitConstants(params, dispatchData.stage_final);
            cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN_BSV32", 1));
            cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
            auto entry_point = GetEntryPoint(finalKernelName, params.layerID, params, entry_part_id++);
            auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
            auto& kernel = kd.kernels[1];
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
            kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
            if (params.mvnNormalizeVariance) {
                kernel.params.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
            }
        }
        kd.internalBufferDataType = Datatype::F32;
    }

    return {kd};
}

KernelsData MVNKernel_bs_fs_yx_bsv32::GetKernelsData(const Params& params) const {
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto slm_per_sg = fsv * 4;
    auto max_lws = params.engineInfo.maxWorkGroupSize;
    auto items_num = orgParams.outputs[0].X().v * orgParams.outputs[0].Y().v * orgParams.outputs[0].Z().v;

    auto enough_slm = max_lws / simd * simd * slm_per_sg <= max_slm;
    auto enough_lws = max_lws / simd >= 1;
    auto enough_items = items_num >= max_lws / simd * simd * pref_work_groups;

    return GetMultiStageKernelsData(orgParams,  enough_slm && enough_lws && enough_items);
}

KernelsPriority MVNKernel_bs_fs_yx_bsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_4;
}
}  // namespace kernel_selector
