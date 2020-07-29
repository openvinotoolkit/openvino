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

#include "fully_connected_kernel_fs_byx_fsv32.h"
#include <utility>
#include <algorithm>

namespace kernel_selector {

static constexpr size_t subGroupSize = 16;
static constexpr size_t outputBlockSizeF = 32;
static constexpr size_t outputBlockSizeB = 4;
static constexpr size_t preferredWGHeight = 4;

ParamsKey FullyConnected_fs_byx_fsv32::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::fs_b_yx_fsv32);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

FullyConnected_fs_byx_fsv32::Parent::DispatchData FullyConnected_fs_byx_fsv32::SetDefault(
    const fully_connected_params& params,
    int autoTuneIndex) const {
    auto runInfo = Parent::SetDefault(params, autoTuneIndex);

    auto blockSizeB = std::min(outputBlockSizeB, params.output.Batch().v);
    auto blockNumB = CeilDiv(params.output.Batch().v, blockSizeB);
    auto wgHeight = std::min(preferredWGHeight, blockNumB);

    runInfo.gws0 = CeilDiv(params.output.Feature().v, outputBlockSizeF);
    runInfo.gws1 = RoundUp(blockNumB, wgHeight);
    runInfo.gws2 = subGroupSize;

    runInfo.lws0 = 1;
    runInfo.lws1 = wgHeight;
    runInfo.lws2 = subGroupSize;

    runInfo.efficiency = FORCE_PRIORITY_5;

    return runInfo;
}

JitConstants FullyConnected_fs_byx_fsv32::GetJitConstants(const fully_connected_params& params,
                                                          const DispatchData& kd) const {
    auto jit = Parent::GetJitConstants(params, kd);

    auto blockSizeB = std::min(outputBlockSizeB, params.output.Batch().v);
    auto blockNumB = CeilDiv(params.output.Batch().v, blockSizeB);
    auto wgHeight = std::min(preferredWGHeight, blockNumB);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("WG_HEIGHT", wgHeight));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_SIZE_B", blockSizeB));

    return jit;
}

KernelsData FullyConnected_fs_byx_fsv32::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    DataLayout::fs_b_yx_fsv32,
                                                    WeightsLayout::os_iyx_osv32__ai32,
                                                    FORCE_PRIORITY_5,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}
}  // namespace kernel_selector
