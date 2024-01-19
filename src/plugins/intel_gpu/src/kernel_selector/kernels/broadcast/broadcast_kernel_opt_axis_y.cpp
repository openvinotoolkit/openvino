// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "broadcast_kernel_opt_axis_y.h"
#include <vector>
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey BroadcastKernelOptAxisY::GetSupportedKey() const {
    ParamsKey k;

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bfwzyx);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();

    k.EnableDynamicShapesSupport();

    return k;
}

KernelsData BroadcastKernelOptAxisY::GetKernelsData(const Params& params, const optional_params& options) const {
    if (!Validate(params, options)) {
        return {};
    }
    return GetCommonKernelsData(params, options);
}

KernelsPriority BroadcastKernelOptAxisY::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_6;
}

JitConstants BroadcastKernelOptAxisY::GetJitConstants(const broadcast_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    jit.AddConstants({MakeJitConstant("BROADCAST_ORDER", params.input_order)});
    jit.AddConstants({MakeJitConstant("VEC_SIZE", vec_size)});
    jit.AddConstants({MakeJitConstant("Y_BLOCKS", y_blocks)});

    return jit;
}

BroadcastKernelBase::DispatchData BroadcastKernelOptAxisY::SetDefault(const broadcast_params& params) const {
    const auto& output = params.outputs[0];
    const auto& input = params.inputs[0];

    DispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();

    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X },
                                                                     { Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                       Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    // Use block calculation only when y is broadcast axis ans x dim is bigger than vec_size
    if ((input.X().v == output.X().v) && (input.X().v >= vec_size) && (input.Y().v != output.Y().v) && (output.Y().v > y_blocks)) {
        dispatchData.gws = { (output.X().v / vec_size), (output.Y().v / y_blocks),
                                output.Z().v * output.W().v * output.Feature().v * output.Batch().v };
    } else {
        dispatchData.gws = { output.X().v, output.Y().v, output.Z().v * output.W().v * output.Feature().v * output.Batch().v };
    }
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    return dispatchData;
}

bool BroadcastKernelOptAxisY::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::BROADCAST ||
        o.GetType() != KernelType::BROADCAST) {
        return false;
    }

    const broadcast_params& params = static_cast<const broadcast_params&>(p);
    if (params.outputs[0].GetDims().size() < 4 || params.outputs[0].GetDims().size() > 6)
        return false;

    return true;
}
}  // namespace kernel_selector
