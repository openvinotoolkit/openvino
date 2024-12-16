// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rope_kernel_opt.h"
#include "kernel_selector_utils.h"
#include <string>

namespace kernel_selector {
ParamsKey RoPEKernelOpt::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

RoPEKernelBase::DispatchData RoPEKernelOpt::SetDefault(const rope_params& params) const {
    DispatchData dispatchData;
    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {
        {Tensor::DataChannelName::BATCH},
        {Tensor::DataChannelName::FEATURE},
        {Tensor::DataChannelName::Y, Tensor::DataChannelName::X}};

    size_t vec_size = GetVecSize(params);
    if (params.is_qwen) {
        auto count = params.head_cnt * std::max(params.rotary_ndims / 2ul, params.head_size - params.rotary_ndims);
        dispatchData.gws = {input.Batch().v, input.Feature().v, count / vec_size};
    } else if (params.is_chatglm) {
        if (params.support_2d_rope) {
            // input  [batch_size, seq_length]
            // output [batch_size, head_count, seq_length, half_rotary_ndims]
            dispatchData.gws = {input.Batch().v * params.head_cnt,
                                input.Feature().v,
                                params.rotary_ndims / 2ul / vec_size};
        } else {
            dispatchData.gws = {input.Batch().v,
                                input.Feature().v,
                                params.head_cnt * (params.rotary_ndims / 2ul / vec_size)};
        }
    } else {
        dispatchData.gws = {output.Batch().v, output.Feature().v, output.Y().v * params.rotary_ndims / 2ul / vec_size};
    }

    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws,
                                                     params.engineInfo,
                                                     input.GetLayout(),
                                                     output.GetLayout(),
                                                     dims_by_gws);

    return dispatchData;
}

JitConstants RoPEKernelOpt::GetJitConstants(const rope_params& params, RoPEKernelBase::DispatchData dispatchData) const {
    JitConstants jit = RoPEKernelBase::GetJitConstants(params, dispatchData);

    jit.AddConstant(MakeJitConstant("VEC_SIZE", GetVecSize(params)));
    return jit;
}

size_t RoPEKernelOpt::GetVecSize(const rope_params& params) const {
    const auto& input = params.inputs[0];
    size_t vec_size = 1;
    switch (input.GetDType()) {
    case Datatype::F16:
        vec_size = 16;
        break;
    case Datatype::F32:
        vec_size = 8;
        break;
    default:
        vec_size = 1;
        break;
    }
    if (params.rotary_ndims % (2 * vec_size) != 0)
        vec_size = 1;

    if (params.is_qwen) {
        auto count = params.head_cnt * std::max(params.rotary_ndims / 2ul, params.head_size - params.rotary_ndims);
        if (count % vec_size != 0)
            vec_size = 1;
    }

    return vec_size;
}

KernelsData RoPEKernelOpt::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority RoPEKernelOpt::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_8;
}
}  // namespace kernel_selector
