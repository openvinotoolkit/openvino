// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "eltwise_kernel_mixed_byxf_and_fs_b_yx_fsv32.h"
#include "kernel_selector_utils.h"
#include <string>
#include <memory>
#include <vector>

namespace kernel_selector {

// TODO: [blocked_formats] does fp32 work well with kernel?
ParamsKey EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableEltwiseBroadcast();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

JitConstants EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetJitConstants(const eltwise_params& params) const {
    return GetJitConstantsCommon(params, false);
}

bool EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::Validate(const Params& params) const {
    if (!EltwiseKernelBase::Validate(params)) {
        return false;
    }

    const auto& ewParams = static_cast<const eltwise_params&>(params);

    const auto& inputs = ewParams.inputs;
    if (inputs.size() != 2) {
        return false;
    }

    for (auto in : inputs) {
        if (in.GetLayout() != DataLayout::fs_b_yx_fsv32 && in.GetLayout() != DataLayout::byxf)
            return false;
    }

    const auto& input1 = inputs[0];
    const auto& input2 = inputs[1];

    if (input1.Feature().v % 32 != 0 || input2.Feature().v % 32 != 0) {
        return false;
    }

    return true;
}

KernelsData EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetKernelsData(const Params& params) const {
    if (!Validate(params)) {
        return {};
    }

    KernelData kd = KernelData::Default<eltwise_params>(params);
    eltwise_params& newParams = *static_cast<eltwise_params*>(kd.params.get());

    std::pair<std::string, std::string> jit;

    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params);

    try {
        auto cldnn_jit = GetJitConstants(newParams);
        cldnn_jit.RemoveConstant("INPUT_0_0");
        cldnn_jit.RemoveConstant("INPUT_0_1");

        cldnn_jit.AddConstants({
            MakeJitConstant("INPUT_0_0", "tmp_input_0"),
            MakeJitConstant("INPUT_0_1", "tmp_input_1"),
            });

        auto input0 = newParams.inputs[0];
        std::vector<size_t> inp0_bfyx = { input0.Batch().v, input0.Feature().v, input0.Y().v, input0.X().v };
        auto input1 = newParams.inputs[1];
        std::vector<size_t> inp1_bfyx = { input1.Batch().v, input1.Feature().v, input1.Y().v, input1.X().v };
        std::vector<std::string> bfyx_str   = { "b", "f0", "y", "x" };
        std::vector<std::string> dims_names = { "BATCH_NUM", "FEATURE_NUM", "SIZE_Y", "SIZE_X" };
        for (size_t dim = 0; dim < inp0_bfyx.size(); dim++) {
            std::string dim_str = bfyx_str[dim];
            std::string jit_str_inp0 = dim_str;
            std::string jit_str_inp1 = dim_str;
            if (inp0_bfyx[dim] > inp1_bfyx[dim]) {
                jit_str_inp1 += " % INPUT1_" + dims_names[dim];
            } else if (inp0_bfyx[dim] < inp1_bfyx[dim]) {
                jit_str_inp0 += " % INPUT0_" + dims_names[dim];
            }
            cldnn_jit.AddConstants({
                MakeJitConstant("INPUT0_DIM_" + dim_str, jit_str_inp0),
                MakeJitConstant("INPUT1_DIM_" + dim_str, jit_str_inp1)
                });
        }

        jit = CreateJit(kernelName, cldnn_jit, entry_point);
    } catch (const std::runtime_error&) {
        return KernelsData();
    }

    auto& kernel = kd.kernels[0];
    size_t x;
    size_t y;
    size_t batches;
    size_t featuresRoundedUp;

    auto dims = newParams.outputs[0].LogicalDims();
    if (newParams.outputs[0].GetLayout() == DataLayout::fs_b_yx_fsv32) {
        x = dims[0];
        y = dims[1];
        batches = dims[2];
        featuresRoundedUp = (((dims[3] - 1) / 32) + 1) * 32;
    } else {  // byxf
        featuresRoundedUp = (((dims[0] - 1) / 32) + 1) * 32;
        x = dims[1];
        y = dims[2];
        batches = dims[3];
    }

    // in fs_b_yx_fsv32 format we will process 2 features per work item, so reads/writes are done in full writes for
    // fp16
    kernel.params.workGroups.global = {x, y, (featuresRoundedUp * batches) / 2};

    kernel.params.workGroups.local = {1, 1, 16};

    kernel.code.kernelString = GetKernelString(kernelName, jit, entry_point, params.engineInfo, EXE_MODE_DEFAULT);
    kernel.params.arguments = GetArgsDesc((uint32_t)newParams.inputs.size(), false, false);

    return {kd};
}

KernelsPriority EltwiseKernel_mixed_byxf_and_fs_b_yx_fsv32::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const eltwise_params&>(params);

    if ((p.outputs[0].GetLayout() == p.inputs[0].GetLayout()) &&
        (p.outputs[0].GetLayout() ==
         p.inputs[1].GetLayout())) {  // There is no need for reordering kernel, better use something more optimal
        return FORCE_PRIORITY_9;
    } else {  // There is need for byxf/fsv32 reordering kernel use this one
        return FORCE_PRIORITY_2;
    }
}
}  // namespace kernel_selector
