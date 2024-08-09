// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reduce_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <string>
#include "common_tools.h"

namespace kernel_selector {
ParamsKey ReduceKernelRef::GetSupportedKey() const {
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
    k.EnableAllInputLayout();
    k.EnableAllOutputLayout();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableDifferentTypes();
    k.EnableDynamicShapesSupport();
    return k;
}

static bool IsScalarOutput(const reduce_params& params) {
    if (params.inputs.size() == 1 && params.outputs.size() == 1
        && params.inputs[0].SimpleLayout() && params.outputs[0].SimpleLayout()
        && params.inputs[0].LogicalSize() >= params.engineInfo.maxWorkGroupSize
        && params.outputs[0].LogicalSize() == 1
        && (params.reduceMode == ReduceMode::SUM || params.reduceMode == ReduceMode::PROD
            || params.reduceMode == ReduceMode::MIN || params.reduceMode == ReduceMode::MAX
            ||params.reduceMode == ReduceMode::MEAN)) {
        auto& in_dims = params.inputs[0].GetDims();
        auto& out_dims = params.outputs[0].GetDims();
        auto has_padding = std::count_if(in_dims.cbegin(), in_dims.cend(),
            [](Tensor::Dim d) {
                return d.pad.before > 0 || d.pad.after > 0;
            }) > 0;
        has_padding |= std::count_if(out_dims.cbegin(), out_dims.cend(),
            [](Tensor::Dim d) {
                return d.pad.before > 0 || d.pad.after > 0;
            }) > 0;
        return !has_padding;
    }
    return false;
}

CommonDispatchData ReduceKernelRef::SetDefault(const reduce_params& params) const {
    CommonDispatchData dispatchData;
    auto in_layout = params.inputs[0].GetLayout();
    auto out_layout = params.outputs[0].GetLayout();
    std::vector<std::vector<Tensor::DataChannelName>> dims_by_gws = {{ Tensor::DataChannelName::X, Tensor::DataChannelName::Y },
                                                                     { Tensor::DataChannelName::Z, Tensor::DataChannelName::W,
                                                                       Tensor::DataChannelName::U, Tensor::DataChannelName::V },
                                                                     { Tensor::DataChannelName::FEATURE, Tensor::DataChannelName::BATCH }};

    dispatchData.gws = { params.outputs[0].X().v * params.outputs[0].Y().v,
                         params.outputs[0].Z().v * params.outputs[0].W().v * params.outputs[0].U().v * params.outputs[0].V().v,
                         params.outputs[0].Batch().v * params.outputs[0].Feature().v };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo, in_layout, out_layout, dims_by_gws);

    if (IsScalarOutput(params)) {
        dispatchData.gws = {params.engineInfo.maxWorkGroupSize, 1, 1};
        dispatchData.lws = dispatchData.gws;
    }
    return dispatchData;
}

JitConstants ReduceKernelRef::GetJitConstants(const reduce_params& params) const {
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

    if (IsScalarOutput(params)) {
        jit.AddConstant(MakeJitConstant("SCALAR_OUTPUT", 1));
        jit.AddConstant(MakeJitConstant("NUM_BLOCKS", params.engineInfo.maxWorkGroupSize));
        jit.AddConstant(MakeJitConstant("BLOCK_STRIDE", params.engineInfo.maxWorkGroupSize));
        jit.AddConstant(MakeJitConstant("TOTAL_NUM_ELEMENTS", params.inputs[0].LogicalSize()));
    }

    return jit;
}

KernelsData ReduceKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ReduceKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
