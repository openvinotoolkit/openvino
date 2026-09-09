// Copyright (C) 2018-2026 Intel Corporation
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
    k.EnableEltwiseBroadcast();
    return k;
}

bool ReduceKernelRef::Validate(const Params& p) const {
    if (!ReduceKernelBase::Validate(p)) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& params = static_cast<const reduce_params&>(p);
    if (!params.weighted) {
        return true;
    }

    if (params.is_shape_agnostic || params.inputs.size() != 2 || params.outputs.size() != 1 || params.reduceMode != ReduceMode::SUM ||
        params.reduceAxes.size() != 1 || params.reduceAxes[0] != 2) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    const auto& values = params.inputs[0];
    const auto& weights = params.inputs[1];
    const auto& output = params.outputs[0];
    if (values.Dimentions() != 4 || weights.Dimentions() != 4 || output.Dimentions() != 4 ||
        (values.GetDType() != Datatype::F16 && values.GetDType() != Datatype::F32) || values.GetDType() != weights.GetDType() || values.X().v != 16 ||
        weights.X().v != 16 || output.X().v != 1 || values.Y().v <= 1024 || values.Batch().v != weights.Batch().v || values.Batch().v != output.Batch().v ||
        weights.Feature().v != 1 || values.Feature().v <= 1 || values.Feature().v != output.Feature().v || values.Y().v != weights.Y().v ||
        values.Y().v != output.Y().v) {
        DO_NOT_USE_THIS_KERNEL(p.layerID);
    }

    return true;
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

    return jit;
}

KernelsData ReduceKernelRef::GetKernelsData(const Params& params) const {
    return GetCommonKernelsData(params);
}

KernelsPriority ReduceKernelRef::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}
}  // namespace kernel_selector
