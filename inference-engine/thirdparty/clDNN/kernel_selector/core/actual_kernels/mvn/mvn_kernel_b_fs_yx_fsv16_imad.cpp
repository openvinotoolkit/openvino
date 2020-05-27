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
    auto kd = Parent::SetDefault(params);

    auto items_num = params.output.X().v * params.output.Y().v * params.output.Z().v;
    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;

    auto max_lws = std::min(max_wg, max_sgs * simd);

    auto lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;

    kd.gws0 = lws;
    kd.gws1 = CeilDiv(params.output.Feature().v, fsv);
    kd.gws2 = params.output.Batch().v;

    kd.lws0 = lws;
    kd.lws1 = 1;
    kd.lws2 = 1;

    kd.itemsNum = 1;

    return kd;
}

JitConstants MVNKernel_b_fs_yx_fsv16_imad::GetJitConstants(const mvn_params& params, DispatchData kd) const {
    auto jits = Parent::GetJitConstants(params, kd);

    auto activation_dt = GetActivationType(params);
    jits.Merge(MakeTypeJitConstants(activation_dt, "MEAN"));
    jits.AddConstant(MakeJitConstant("SIMD", simd));
    jits.AddConstant(MakeJitConstant("LWS", kd.lws0));
    jits.AddConstant(MakeJitConstant("GWS", kd.gws0));
    jits.AddConstant(MakeJitConstant("ITEM_GROUPS", kd.itemsNum));

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
    MultiDispatchData md;

    auto items_num = params.output.X().v * params.output.Y().v * params.output.Z().v;
    auto max_wg = params.engineInfo.maxWorkGroupSize;
    auto slm_per_sg = fsv * 4;
    auto max_slm = params.engineInfo.maxLocalMemSize;
    auto max_sgs = max_slm / slm_per_sg;

    auto max_lws = std::min(max_wg, max_sgs * simd);
    auto lws = std::max(std::min(items_num, max_lws) / simd, (size_t)1) * simd;

    // TODO Check if larger number of work-groups does not provide benefit
    size_t item_groups = pref_work_groups;
    md.item_groups = item_groups;

    size_t stage1_lws = lws;

    md.stage_1.gws0 = stage1_lws * item_groups;
    md.stage_1.gws1 = CeilDiv(params.output.Feature().v, fsv);
    md.stage_1.gws2 = params.output.Batch().v;

    md.stage_1.lws0 = stage1_lws;
    md.stage_1.lws1 = 1;
    md.stage_1.lws2 = 1;

    md.stage_1.itemsNum = item_groups;

    size_t stage2_lws = std::max(std::min(item_groups, max_lws) / simd, (size_t)1) * simd;

    md.stage_2.gws0 = stage2_lws;
    md.stage_2.gws1 = CeilDiv(params.output.Feature().v, fsv);
    md.stage_2.gws2 = params.output.Batch().v;

    md.stage_2.lws0 = stage2_lws;
    md.stage_2.lws1 = 1;
    md.stage_2.lws2 = 1;

    md.stage_2.itemsNum = item_groups;

    md.stage_final.gws0 = std::max(items_num / simd, (size_t)1) * simd;
    md.stage_final.gws1 = CeilDiv(params.output.Feature().v, fsv);
    md.stage_final.gws2 = params.output.Batch().v;

    md.stage_final.lws0 = simd;
    md.stage_final.lws1 = 1;
    md.stage_final.lws2 = 1;

    md.stage_final.itemsNum = 1;

    return md;
}

KernelsData MVNKernel_b_fs_yx_fsv16_imad::GetMultiStageKernelsData(const mvn_params& params,
                                                                   const optional_params& options,
                                                                   float estimated_time) const {
    if (!Validate(params, options))
        return {};

    constexpr size_t intermidiate_bytes = 4;
    const mvn_params& orgParams = static_cast<const mvn_params&>(params);

    auto runInfo = SetDefaultForMulti(orgParams);

    size_t kernels_num = params.mvnNormalizeVariance ? 5 : 3;
    KernelData kd = KernelData::Default<mvn_params>(params, kernels_num);

    auto finalKernelName = GetKernelName(orgParams);
    {
        // Mean first stage
        auto cldnn_jit = GetJitConstants(orgParams, runInfo.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[0];
        FillCLKernelData(kernel,
                         runInfo.stage_1,
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
                                         runInfo.item_groups * intermidiate_bytes);
    }
    {
        // Mean second stage
        auto cldnn_jit = GetJitConstants(orgParams, runInfo.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MEAN_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[1];
        FillCLKernelData(kernel,
                         runInfo.stage_2,
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
        auto cldnn_jit = GetJitConstants(orgParams, runInfo.stage_1);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_1", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[2];
        FillCLKernelData(kernel,
                         runInfo.stage_1,
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
        auto cldnn_jit = GetJitConstants(orgParams, runInfo.stage_2);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_VAR_2", 1));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[3];
        FillCLKernelData(kernel,
                         runInfo.stage_2,
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
        auto cldnn_jit = GetJitConstants(orgParams, runInfo.stage_final);
        cldnn_jit.AddConstant(MakeJitConstant("MVN_KERNEL_MAIN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_MEAN", 1));
        cldnn_jit.AddConstant(MakeJitConstant("PRECALC_VARIANCE", params.mvnNormalizeVariance));
        auto entry_point = GetEntryPoint(finalKernelName, orgParams.layerID, options);
        auto jit = CreateJit(finalKernelName, cldnn_jit, entry_point);
        auto& kernel = kd.kernels[kernels_num - 1];
        FillCLKernelData(kernel,
                         runInfo.stage_final,
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
