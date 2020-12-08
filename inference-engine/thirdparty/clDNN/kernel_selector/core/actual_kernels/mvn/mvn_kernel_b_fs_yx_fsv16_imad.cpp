// Copyright (c) 2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mvn_kernel_b_fs_yx_fsv16_imad.hpp"
#include "common/common_tools.h"

#include <string>
#include <algorithm>
#include <iostream>

namespace kernel_selector {

static constexpr size_t simd = 16;
static constexpr size_t fsv = 16;
static constexpr size_t pref_work_groups = 16;

ParamsKey MVNKernel_b_fs_yx_fsv16_imad::GetSupportedKey() const {
    ParamsKey k;

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
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableDifferentTypes();
    k.EnableBatching();
    // TODO Add support for across channels
    // k.EnableMVNMode(MVNMode::ACROSS_CHANNELS);
    k.EnableMVNMode(MVNMode::WITHIN_CHANNELS);
    k.EnableMVNNormalizeVariance();

    return k;
}

bool MVNKernel_b_fs_yx_fsv16_imad::Validate(const Params& p, const optional_params& options) const {
    if (!Parent::Validate(p, options))
        return false;

    auto params = static_cast<const mvn_params&>(p);

    // TODO Add support for input padding via iterating over y (parallel or in kernel).
    if (params.inputs[0].X().pad.Total() != 0 || params.inputs[0].Y().pad.Total() != 0 ||
        params.inputs[0].Z().pad.Total() != 0)
        return false;

    return true;
}

MVNKernelBase::DispatchData MVNKernel_b_fs_yx_fsv16_imad::SetDefault(const mvn_params& params) const {
    auto dispatchData = Parent::SetDefault(params);

    auto items_num = params.output.X().v * params.output.Y().v * params.output.Z().v;
    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;

    auto max_lws = std::min(max_wg, max_sgs * simd);

    auto lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;

    dispatchData.gws[0] = lws;
    dispatchData.gws[1] = CeilDiv(params.output.Feature().v, fsv);
    dispatchData.gws[2] = params.output.Batch().v;

    dispatchData.lws[0] = lws;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    dispatchData.itemsNum = 1;

    return dispatchData;
}

JitConstants MVNKernel_b_fs_yx_fsv16_imad::GetJitConstants(const mvn_params& params, DispatchData dispatchData) const {
    auto jits = Parent::GetJitConstants(params, dispatchData);

    auto activation_dt = GetActivationType(params);
    jits.Merge(MakeTypeJitConstants(activation_dt, "MEAN"));
    jits.AddConstant(MakeJitConstant("SIMD", simd));
    jits.AddConstant(MakeJitConstant("LWS", dispatchData.lws[0]));
    jits.AddConstant(MakeJitConstant("GWS", dispatchData.gws[0]));
    jits.AddConstant(MakeJitConstant("ITEM_GROUPS", dispatchData.itemsNum));

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
        jits.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }
    return jits;
}

MVNKernel_b_fs_yx_fsv16_imad::MultiDispatchData MVNKernel_b_fs_yx_fsv16_imad::SetDefaultForMulti(
    const mvn_params& params) const {
    MultiDispatchData dispatchData;

    auto items_num = params.output.X().v * params.output.Y().v * params.output.Z().v;
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

    dispatchData.stage_1.gws[0] = stage1_lws * item_groups;
    dispatchData.stage_1.gws[1] = CeilDiv(params.output.Feature().v, fsv);
    dispatchData.stage_1.gws[2] = params.output.Batch().v;

    dispatchData.stage_1.lws[0] = stage1_lws;
    dispatchData.stage_1.lws[1] = 1;
    dispatchData.stage_1.lws[2] = 1;

    dispatchData.stage_1.itemsNum = item_groups;

    size_t stage2_lws = std::max(std::min(item_groups, max_lws) / simd, (size_t)1) * simd;

    dispatchData.stage_2.gws[0] = stage2_lws;
    dispatchData.stage_2.gws[1] = CeilDiv(params.output.Feature().v, fsv);
    dispatchData.stage_2.gws[2] = params.output.Batch().v;

    dispatchData.stage_2.lws[0] = stage2_lws;
    dispatchData.stage_2.lws[1] = 1;
    dispatchData.stage_2.lws[2] = 1;

    dispatchData.stage_2.itemsNum = item_groups;

    dispatchData.stage_final.gws[0] = std::max(items_num / simd, (size_t)1) * simd;
    dispatchData.stage_final.gws[1] = CeilDiv(params.output.Feature().v, fsv);
    dispatchData.stage_final.gws[2] = params.output.Batch().v;

    dispatchData.stage_final.lws[0] = simd;
    dispatchData.stage_final.lws[1] = 1;
    dispatchData.stage_final.lws[2] = 1;

    dispatchData.stage_final.itemsNum = 1;

    return dispatchData;
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetMultiStageKernelsData(const mvn_params& params,
                                                                   const optional_params& options,
                                                                   float estimated_time) const {
    if (!Validate(params, options))
        return {};

    constexpr size_t intermidiate_bytes = 4;
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    auto dispatchData = SetDefaultForMulti(orgParams);

    size_t kernels_num = params.mvnNormalizeVariance ? 5 : 3;
    KernelData kd = KernelData::Default<mvn_params>(params, kernels_num);

    auto finalKernelName = GetKernelName(orgParams);
    {
        // Mean first stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
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
        kernel.arguments.clear();  // Clear original output argument
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kd.internalBufferSizes.push_back(params.output.Batch().v * Align(params.output.Feature().v, fsv) *
                                         dispatchData.item_groups * intermidiate_bytes);
    }
    {
        // Mean second stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
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
        kernel.arguments.clear();  // Clear original output argument
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kd.internalBufferSizes.push_back(params.output.Batch().v * Align(params.output.Feature().v, fsv) *
                                         intermidiate_bytes);
    }
    if (params.mvnNormalizeVariance) {
        // Variance first stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
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
        kernel.arguments.clear();  // Clear original output argument
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 0});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
    }
    if (params.mvnNormalizeVariance) {
        // Variance second stage
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
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
        kernel.arguments.clear();  // Clear original output argument
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 0});
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        kd.internalBufferSizes.push_back(params.output.Batch().v * Align(params.output.Feature().v, fsv) *
                                         intermidiate_bytes);
    }
    {  // Final
        auto cldnn_jit = GetJitConstants(orgParams, dispatchData.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_MEAN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
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
        kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 1});
        if (params.mvnNormalizeVariance) {
            kernel.arguments.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 2});
        }
    }
    kd.internalBufferDataType = Datatype::F32;
    kd.estimatedTime = estimated_time;

    return {kd};
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetKernelsData(const Params& params, const optional_params& optParams) const {
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto slm_per_sg = fsv * 4;
    auto max_lws = params.engineInfo.maxWorkGroupSize;
    auto items_num = orgParams.output.X().v * orgParams.output.Y().v * orgParams.output.Z().v;

    auto enough_slm = max_lws / simd * simd * slm_per_sg <= max_slm;
    auto enough_lws = max_lws / simd >= 1;
    auto enough_items = items_num >= max_lws / simd * simd * pref_work_groups;

    if (enough_slm && enough_lws && enough_items)
        return GetMultiStageKernelsData(orgParams, optParams, FORCE_PRIORITY_4);
    else
        return GetCommonKernelsData(params, optParams, FORCE_PRIORITY_4);
}
}  // namespace kernel_selector
