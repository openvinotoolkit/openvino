// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_simple_to_scalar.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {
ParamsKey ReduceKernelSimpleToScalar::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::byfx);
    k.EnableInputLayout(DataLayout::bxfy);
    k.EnableInputLayout(DataLayout::fbyx);
    k.EnableInputLayout(DataLayout::fyxb);
    k.EnableInputLayout(DataLayout::bfxy);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::bzyxf);
    k.EnableInputLayout(DataLayout::bfwzyx);
    k.EnableInputLayout(DataLayout::bfuwzyx);
    k.EnableInputLayout(DataLayout::bfvuwzyx);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::byfx);
    k.EnableOutputLayout(DataLayout::bxfy);
    k.EnableOutputLayout(DataLayout::fbyx);
    k.EnableOutputLayout(DataLayout::fyxb);
    k.EnableOutputLayout(DataLayout::bfxy);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bzyxf);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableOutputLayout(DataLayout::bfuwzyx);
    k.EnableOutputLayout(DataLayout::bfvuwzyx);

    k.EnableBatching();
    k.EnableDifferentTypes();

    return k;
}

bool ReduceKernelSimpleToScalar::Validate(const Params& p) const {
    const reduce_params& params = static_cast<const reduce_params&>(p);

    if (params.inputs.size() != 1 || params.outputs.size() != 1)
        return false;

    if (!params.inputs[0].SimpleLayout() || !params.outputs[0].SimpleLayout())
        return false;

    if (params.inputs[0].LogicalSize() < params.engineInfo.maxWorkGroupSize)
        return false;

    if (params.outputs[0].LogicalSize() != 1)
        return false;

    std::set<ReduceMode> supported_modes = {
        ReduceMode::SUM,
        ReduceMode::PROD,
        ReduceMode::MIN,
        ReduceMode::MAX,
        ReduceMode::MEAN,
    };

    if (supported_modes.find(params.reduceMode) == supported_modes.end())
        return false;

    auto& in_dims = params.inputs[0].GetDims();
    auto& out_dims = params.outputs[0].GetDims();
    auto has_padding = std::count_if(in_dims.cbegin(), in_dims.cend(),
        [](Tensor::Dim d) { return d.pad.before > 0 || d.pad.after > 0; }) > 0;
    has_padding |= std::count_if(out_dims.cbegin(), out_dims.cend(),
        [](Tensor::Dim d) {
            return d.pad.before > 0 || d.pad.after > 0;
        }) > 0;

    if (has_padding)
        return false;

    return true;
}

CommonDispatchData ReduceKernelSimpleToScalar::SetDefault(const reduce_params& params) const {
    CommonDispatchData dispatchData;
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                       Tensor::DataChannelName::U, Tensor::DataChannelName::V },
                                                                     { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    dispatchData.gws = {params.engineInfo.maxWorkGroupSize, 1, 1};
    dispatchData.lws = dispatchData.gws;
    return dispatchData;
}

JitConstants ReduceKernelSimpleToScalar::GetJitConstants(const reduce_params& params) const {
    auto jit = ReduceKernelBase::GetJitConstants(params);

    jit.Merge(MakeTypeJitConstants(GetActivationType(params), "ACTIVATION"));
    jit.Merge(MakeTypeJitConstants(GetAccumulatorType(params), "ACCUMULATOR"));
    jit.Merge(MakeTypeJitConstants(GetFinalAccumulatorType(params), "FINAL_ACCUMULATOR"));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);

        std::vector<std::string> idx_order;
        switch (DataTensor::ChannelsCount(params.inputs[0].GetLayout())) {
            case 8: idx_order = {"b", "f", "v", "u", "w", "z", "y", "x" }; break;
            case 7: idx_order = {"b", "f", "u", "w", "z", "y", "x" }; break;
            case 6: idx_order = {"b", "f", "w", "z", "y", "x" }; break;
            case 5: idx_order = {"b", "f", "z", "y", "x" }; break;
            default: idx_order = {"b", "f", "y", "x" }; break;
        }

        FusedOpsConfiguration conf = {"",
                                      idx_order,
                                      "reduce_result",
                                      input_dt,
                                      1,
                                      LoadType::LT_UNALIGNED,
                                      BoundaryCheck::DISABLED,
                                      IndexType::TENSOR_COORD,
                                      Tensor::DataChannelName::X};

        jit.Merge(MakeFusedOpsJitConstants(params, {conf}));
    }

    jit.AddConstant(MakeJitConstant("NUM_BLOCKS", params.engineInfo.maxWorkGroupSize));
    jit.AddConstant(MakeJitConstant("BLOCK_STRIDE", params.engineInfo.maxWorkGroupSize));
    jit.AddConstant(MakeJitConstant("TOTAL_NUM_ELEMENTS", params.inputs[0].LogicalSize()));

    return jit;
}

KernelsData ReduceKernelSimpleToScalar::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ReduceKernelSimpleToScalar::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}
}  // namespace kernel_selector
