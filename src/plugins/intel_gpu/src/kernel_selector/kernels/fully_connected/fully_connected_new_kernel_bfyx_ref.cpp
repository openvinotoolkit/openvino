// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "fully_connected_new_kernel_bfyx_ref.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {
ParamsKey FullyConnectedNew_bfyx_Ref::GetSupportedKey() const {
    ParamsKey k;

    k.EnableNewShapeInfer();

    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableAllInputLayout();
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

FullyConnectedNew_bfyx_Ref::DispatchData FullyConnectedNew_bfyx_Ref::SetDefault(const fully_connected_params& params,
                                                                          int) const {
    auto dispatchData = Parent::SetDefault(params);

    std::vector<size_t> global = {1, 1, 1};

    const auto& output = params.outputs[0];
    const auto rank = params.output_shape.rank().get_length();
    switch (rank) {
        case 1: {
            break;
        }
        case 2: {
            global = { output.Batch().v, output.Feature().v, 1 };
            break;
        }
        case 3: {
            global = { output.Batch().v, output.Feature().v, output.Y().v };
            break;
        }
        case 4: {
            global = { output.Batch().v * output.Feature().v, output.Y().v, output.X().v };
            break;
        }
        case 5: {
            global = { output.Batch().v * output.Feature().v, output.Z().v * output.Y().v, output.X().v };
            break;
        }
        case 6: {
            global = { output.Batch().v * output.Feature().v, output.W().v * output.Z().v, output.Y().v * output.X().v };
            break;
        }
        default: {
            break;
        }
    }

    dispatchData.gws = global;
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsPriority FullyConnectedNew_bfyx_Ref::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants FullyConnectedNew_bfyx_Ref::GetJitConstants(const fully_connected_params& params,
    const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    JitConstants jit = Parent::GetJitConstants(params, dispatchData);

    Datatype accumulator_dt = GetAccumulatorType(params);
    jit.Merge(MakeTypeJitConstants(accumulator_dt, "ACCUMULATOR"));

    const auto rank = params.output_shape.rank().get_length();
    jit.AddConstant(MakeJitConstant("RANK", rank));

    return jit;
}

KernelsData FullyConnectedNew_bfyx_Ref::GetKernelsData(const Params& params, const optional_params& options) const {
    auto& fc_params = static_cast<const fully_connected_params&>(params);
    KernelsData res = {};

    const auto get_weights_layout = [](const fully_connected_params& params) {
        const auto rank = params.weights_shape.rank().get_length();
        switch (rank) {
            case 2:
            case 3:
            case 4: return WeightsLayout::oiyx;
            case 5: return WeightsLayout::oizyx;
            case 6: return WeightsLayout::goizyx;
            default: return WeightsLayout::oiyx;
        }
    };

    const auto input_layout = fc_params.inputs[0].GetLayout();
    const auto weights_layout = get_weights_layout(fc_params);

    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(
            params,
            options,
            input_layout,
            weights_layout,
            static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

bool FullyConnectedNew_bfyx_Ref::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    const auto& fc_params = static_cast<const fully_connected_params&>(params);

    //this kernel supports only new shape inference
    if (!fc_params.new_shape_infer) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
