// Copyright (c) 2019-2020 Intel Corporation
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
#include <vector>

#include "fully_connected_kernel_imad.h"

// IMAD Fully_Connected primitive implementation.
// Limitations are:
// 1. Input=Fx1x1 with Filter=1x1
// 2. No data padding

namespace kernel_selector {
ParamsKey FullyConnectedKernelIMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputWeightsType(WeightsType::INT8);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv4);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);

    k.EnableOutputLayout(DataLayout::bf);

    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    return k;
}

FullyConnectedKernelIMAD::Parent::DispatchData FullyConnectedKernelIMAD::SetDefault(
    const fully_connected_params& params,
    int) const {
    const int simdSize = 16;

    auto dispatchData = Parent::SetDefault(params);

    dispatchData.gws[0] = RoundUp(params.output.Feature().v, simdSize);
    dispatchData.gws[1] = params.output.Batch().v;
    dispatchData.gws[2] = 1;

    dispatchData.lws[0] = simdSize;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    return dispatchData;
}  // SetDefault

bool FullyConnectedKernelIMAD::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options)) {
        return false;
    }

    const auto& newParams = static_cast<const fully_connected_params&>(params);
    const auto& in = newParams.inputs[0];
    const auto& weights = newParams.weights;

    if ((in.X().v != 1) || (in.Y().v != 1) || (weights.X().v != 1) || (weights.Y().v != 1)) {
        // Currently only Input=Fx1x1 with Filter=1x1 is supported
        return false;
    }
    if ((in.X().pad.before != 0) || (in.X().pad.after != 0) || (in.Y().pad.before != 0) || (in.Y().pad.after != 0)) {
        // Padding is not supported
        return false;
    }
    if (in.Feature().v % (4 * 8)) {
        // Algorith requires 4 bytes read as one int
        // with specific weight format os_is_yx_osv16_isv4
        // wich will read 8 elements per reading
        return false;
    }

    return true;
}  // Validate

JitConstants FullyConnectedKernelIMAD::GetJitConstants(const fully_connected_params& params, const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = { "", {"b", "f", "y", "x"}, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnectedKernelIMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    input.GetLayout(),
                                                    WeightsLayout::os_is_yx_osv16_isv4,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}

KernelsPriority FullyConnectedKernelIMAD::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_1;
}
}  // namespace kernel_selector
