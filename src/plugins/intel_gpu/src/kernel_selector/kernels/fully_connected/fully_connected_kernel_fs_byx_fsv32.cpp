// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey FullyConnected_fs_byx_fsv32::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

FullyConnected_fs_byx_fsv32::Parent::DispatchData FullyConnected_fs_byx_fsv32::SetDefault(
    const fully_connected_params& params,
    int autoTuneIndex,
    int /*kernel_number*/) const {
    auto dispatchData = Parent::SetDefault(params, autoTuneIndex);

    auto blockSizeB = std::min(outputBlockSizeB, params.outputs[0].Batch().v);
    auto blockNumB = CeilDiv(params.outputs[0].Batch().v, blockSizeB);
    auto wgHeight = std::min(preferredWGHeight, blockNumB);

    dispatchData.gws[0] = CeilDiv(params.outputs[0].Feature().v, outputBlockSizeF);
    dispatchData.gws[1] = RoundUp(blockNumB, wgHeight);
    dispatchData.gws[2] = subGroupSize;

    dispatchData.lws[0] = 1;
    dispatchData.lws[1] = wgHeight;
    dispatchData.lws[2] = subGroupSize;

    return dispatchData;
}

JitConstants FullyConnected_fs_byx_fsv32::GetJitConstants(const fully_connected_params& params,
                                                          const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    auto blockSizeB = std::min(outputBlockSizeB, params.outputs[0].Batch().v);
    auto blockNumB = CeilDiv(params.outputs[0].Batch().v, blockSizeB);
    auto wgHeight = std::min(preferredWGHeight, blockNumB);

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", subGroupSize));
    jit.AddConstant(MakeJitConstant("WG_HEIGHT", wgHeight));
    jit.AddConstant(MakeJitConstant("OUTPUT_BLOCK_SIZE_B", blockSizeB));

    return jit;
}

bool FullyConnected_fs_byx_fsv32::Validate(const Params& p) const {
    if (!FullyConnectedKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const fully_connected_params&>(p);

    if (!params.bias.empty()) {
        if (params.inputs[0].GetDType() != params.bias[0].GetDType()) {
            return false;
        }
    }

    return true;
}

KernelsData FullyConnected_fs_byx_fsv32::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::fs_b_yx_fsv32,
                                                    WeightsLayout::os_iyx_osv32__ai32,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_fs_byx_fsv32::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
