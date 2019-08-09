// Copyright (c) 2016 Intel Corporation
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

#include "fully_connected_kernel_mmad.h"

namespace kernel_selector {
ParamsKey FullyConnectedKernelMMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableInt8Quantization();
    k.EnableOutputCalibration();
    return k;
}

FullyConnectedKernelMMAD::DispatchData FullyConnectedKernelMMAD::SetDefault(const fully_connected_params& params,
                                                                            int) const {
    auto runInfo = Parent::SetDefault(params);

    constexpr size_t sub_group_size = 8;
    const auto of_maps = params.output.Feature().v;
    const size_t of_threads_per_batch = RoundUp(of_maps, sub_group_size);

    runInfo.gws0 = 1;
    runInfo.gws1 = 1;
    runInfo.gws2 = of_threads_per_batch * params.output.Batch().v;

    runInfo.lws0 = 1;
    runInfo.lws1 = 1;
    runInfo.lws2 = sub_group_size;

    return runInfo;
}

JitConstants FullyConnectedKernelMMAD::GetJitConstants(const fully_connected_params& params,
                                                       const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", runInfo.lws2));

    // pitch for special block format used in this kernel
    const size_t ifm_32_aligned = Align(params.weights.IFM().v, 32);
    const size_t filter_ofm_block_pitch =
        (ifm_32_aligned / 32) * params.weights.X().v * params.weights.Y().v * 4 * 8 * 8;
    jit.AddConstant(MakeJitConstant("FILTER_OFM_BLOCK_PITCH", filter_ofm_block_pitch));

    return jit;
}

KernelsData FullyConnectedKernelMMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    DataLayout::byxf_af32,
                                                    {WeightsLayout::os_is_yx_isa8_osv8_isv4},
                                                    DONT_USE_IF_HAVE_SOMETHING_ELSE,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}
}  // namespace kernel_selector
