// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "batch_to_space_kernel_base.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {

bool BatchToSpaceKernelBase::Validate(const Params& p) const {
    if (p.GetType() != KernelType::BATCH_TO_SPACE) {
        return false;
    }

    const batch_to_space_params& params = static_cast<const batch_to_space_params&>(p);
    for (auto& fused_op : params.fused_ops) {
        if (!IsFusedPrimitiveSupported(fused_op))
            return false;
    }

    if (params.inputs[0].Dimentions() > 6)
        return false;

    return true;
}

CommonDispatchData BatchToSpaceKernelBase::SetDefault(const batch_to_space_params& params) const {
    const auto& out = params.outputs[0];

    CommonDispatchData dispatchData;
    if (out.GetLayout() == DataLayout::b_fs_yx_fsv16 && out.Feature().v % 16 == 0) {
        dispatchData.gws = { out.Batch().v, out.Feature().v, out.Y().v * out.X().v };
        dispatchData.lws = { 1, 16, 1 };
    } else {
        auto in_layout = params.inputs[0].GetLayout();
        auto out_layout = out.GetLayout();
        std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::BATCH },
                                                                         { Tensor::DataChannelName::FEATURE },
                                                                         { Tensor::DataChannelName::X, Tensor::DataChannelName::Y,
                                                                           Tensor::DataChannelName::Z, Tensor::DataChannelName::W }};

        dispatchData.gws = { out.Batch().v, out.Feature().v, out.W().v * out.Z().v * out.Y().v * out.X().v };
        dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);
    }

    return dispatchData;
}

inline std::string GetInputTypeStr(size_t idx) {
    return "INPUT" + std::to_string(idx) + "_TYPE";
}

JitConstants BatchToSpaceKernelBase::GetJitConstants(const batch_to_space_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    auto makeJitConstForParam = [](JitConstants& jit, const std::string name, const DimTensor<uint32_t>& args, const size_t default_value) {
        jit.AddConstant(MakeJitConstant(name + "_SIZES", args));
        jit.AddConstant(MakeJitConstant(name + "_BATCH", args.b));
        jit.AddConstant(MakeJitConstant(name + "_FEATURE", args.f));
        jit.AddConstant(MakeJitConstant(name + "_Y", args.y));
        jit.AddConstant(MakeJitConstant(name + "_X", args.x));

        if (args.w != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", args.w));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else if (args.z != 0) {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", args.z));
        } else {
            jit.AddConstant(MakeJitConstant(name + "_W", default_value));
            jit.AddConstant(MakeJitConstant(name + "_Z", default_value));
        }
    };

    if (params.block_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("BLOCK_TYPE", GetInputTypeStr(params.block_input_index)));
        jit.AddConstant(MakeJitConstant("BLOCK_DIMS", params.block_dims));
    } else {
        makeJitConstForParam(jit, "BLOCK_SHAPE", params.block_shape, 1);
    }

    if (params.begin_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("BEGIN_TYPE", GetInputTypeStr(params.begin_input_index)));
        jit.AddConstant(MakeJitConstant("BEGIN_DIMS", params.begin_dims));
    } else {
        makeJitConstForParam(jit, "CROPS_BEGIN", params.crops_begin, 0);
    }

    if (params.end_type == base_params::ArgType::Input) {
        jit.AddConstant(MakeJitConstant("END_TYPE", GetInputTypeStr(params.end_input_index)));
        jit.AddConstant(MakeJitConstant("END_DIMS", params.end_dims));
    } else {
        makeJitConstForParam(jit, "CROPS_END", params.crops_end, 0);
    }

    return jit;
}

KernelsData BatchToSpaceKernelBase::GetCommonKernelsData(const Params& params) const {
    KernelData kd = KernelData::Default<batch_to_space_params>(params);
    batch_to_space_params& newParams = *static_cast<batch_to_space_params*>(kd.params.get());

    if (!Validate(params)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     "", false, false, static_cast<int>(newParams.inputs.size()),
                     GetFusedPrimitiveInputsCount(params), 1, newParams.is_shape_agnostic);

    return { kd };
}
}  // namespace kernel_selector
