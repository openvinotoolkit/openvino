// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <kernel_selector_utils.h>
#include "arg_max_min_kernel_axis.h"

namespace kernel_selector {

namespace {
size_t getOperationNumber(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::FEATURE: return params.outputs[0].Batch().v * params.outputs[0].Z().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Z: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Y().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::Y: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].X().v;
        case ArgMaxMinAxis::X: return params.outputs[0].Batch().v * params.outputs[0].Feature().v * params.outputs[0].Z().v * params.outputs[0].Y().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

std::string getOperationNumberString(const arg_max_min_params& params) {
    const auto& output = params.outputs[0];
    auto x = toCodeString(output.X(), 11);
    auto y = toCodeString(output.Y(), 10);
    auto z = toCodeString(output.Z(), 9);
    auto w = toCodeString(output.W(), 8);
    auto f = toCodeString(output.Feature(), 7);
    auto b = toCodeString(output.Batch(), 6);
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return toVectorMulString({x, y, z, f});
        case ArgMaxMinAxis::FEATURE: return toVectorMulString({x, y, z, b});
        case ArgMaxMinAxis::Z: return toVectorMulString({y, z, f, b});
        case ArgMaxMinAxis::Y: return toVectorMulString({x, z, f, b});
        case ArgMaxMinAxis::X: return toVectorMulString({y, z, f, b});
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}

size_t getSortSize(const arg_max_min_params& params) {
    switch (params.argMaxMinAxis) {
        case ArgMaxMinAxis::BATCH: return params.inputs[0].Batch().v;
        case ArgMaxMinAxis::FEATURE: return params.inputs[0].Feature().v;
        case ArgMaxMinAxis::Z: return params.inputs[0].Z().v;
        case ArgMaxMinAxis::Y: return params.inputs[0].Y().v;
        case ArgMaxMinAxis::X: return params.inputs[0].X().v;
        default:
            throw std::invalid_argument("Unsupported axis");
    }
}
}  // namespace

ParamsKey ArgMaxMinKernelAxis::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableAllOutputDataType();
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv32);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv32_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::BATCH);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::X);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Y);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::Z);
    k.EnableArgMaxMinAxis(ArgMaxMinAxis::FEATURE);
    k.EnableDifferentTypes();
    k.EnableBatching();
    k.EnableTensorPitches();
    k.EnableTensorOffset();
    k.EnableDynamicShapesSupport();
    return k;
}

bool ArgMaxMinKernelAxis::Validate(const Params& p, const optional_params& o) const {
    if (!ArgMaxMinKernelBase::Validate(p, o)) {
        return false;
    }

    const arg_max_min_params& params = static_cast<const arg_max_min_params&>(p);

    if (params.inputs.size() > 1) {
        if (params.inputs[1].PitchesDifferFromLogicalDims() || params.outputs[0].PitchesDifferFromLogicalDims())
            return false;
    }

    return true;
}

ArgMaxMinKernelBase::DispatchData ArgMaxMinKernelAxis::SetDefault(const arg_max_min_params& params) const {
    DispatchData dispatchData;

    if (!params.has_dynamic_tensors()) {
        size_t ops_size = getOperationNumber(params);
        ops_size = ops_size > 1 ? Align(ops_size, 32) : 1;
        size_t sort_size = params.argMaxMinSortType == ArgMaxMinSortType::VALUE ? getSortSize(params) : 1;

        dispatchData.gws = { ops_size, sort_size, 1 };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);
    }

    return dispatchData;
}

KernelsData ArgMaxMinKernelAxis::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    const arg_max_min_params& orgParams = static_cast<const arg_max_min_params&>(params);

    auto dispatchData = SetDefault(orgParams);
    KernelData kd = KernelData::Default<arg_max_min_params>(params);
    kd.update_dispatch_data_func = [this](const Params& params, KernelData& kd) {
        const auto& prim_params = static_cast<const arg_max_min_params&>(params);
        auto dispatchData = SetDefault(prim_params);
        OPENVINO_ASSERT(kd.kernels.size() == 1, "[GPU] Invalid kernels size for update dispatch data func");
        kd.kernels[0].params.workGroups.global = dispatchData.gws;
        kd.kernels[0].params.workGroups.local = dispatchData.lws;
    };

    auto cldnn_jit = GetJitConstants(orgParams);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     EXE_MODE_DEFAULT,
                     false,
                     false,
                     1,
                     GetFusedPrimitiveInputsCount(params),
                     orgParams.use_multiple_outputs ? 2 : 1,
                     orgParams.outputs[0].is_dynamic());

    if (orgParams.has_second_output && !orgParams.use_multiple_outputs)
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::INPUT, 1});

    return {kd};
}

KernelsPriority ArgMaxMinKernelAxis::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_3;
}

JitConstants ArgMaxMinKernelAxis::GetJitConstants(const arg_max_min_params& params) const {
    auto jit = ArgMaxMinKernelBase::GetJitConstants(params);

    if (params.has_dynamic_tensors()) {
        const std::string gws_0 = "get_global_size(0)";
        const std::string operation_num_comp = "(GWS_0!=1)";
        const std::string operation_num = getOperationNumberString(params);
        jit.AddConstant(MakeJitConstant("GWS_0", gws_0));
        jit.AddConstant(MakeJitConstant("OPERATION_NUM_COMP", operation_num_comp));
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", operation_num));
    } else {
        const size_t operation_num = getOperationNumber(params);
        jit.AddConstant(MakeJitConstant("OPERATION_NUM_COMP", operation_num > 1));
        jit.AddConstant(MakeJitConstant("OPERATION_NUM", operation_num));
    }
    if (params.argMaxMinSortType == ArgMaxMinSortType::VALUE)
        jit.AddConstant(MakeJitConstant("SORT_BY_VALUE", 1));
    else
        jit.AddConstant(MakeJitConstant("SORT_BY_INDEX", 1));

    if (params.has_second_output) {
        jit.AddConstant(MakeJitConstant("SECOND_OUTPUT_EXIST", 1));
        if (params.use_multiple_outputs) {
            jit.AddConstant(MakeJitConstant("MULTIPLE_OUTPUTS", 1));
        }
    }

    if (params.values_first)
        jit.AddConstant(MakeJitConstant("TOP_K_ORDER", 1));

    return jit;
}
}  // namespace kernel_selector
