// Copyright (c) 2016-2020 Intel Corporation
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
#include "kernel_selector_utils.h"

namespace kernel_selector {

namespace {
    static const size_t sub_group_size = 8;
}  // namespace

ParamsKey FullyConnectedKernelMMAD::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableDifferentInputWeightsTypes();
    k.EnableDifferentTypes();

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf_af32);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv32);
    k.EnableOutputLayout(DataLayout::bf);

    k.EnableBiasPerOutput();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    return k;
}

bool FullyConnectedKernelMMAD::Validate(const Params& params, const optional_params& options) const {
    if (!Parent::Validate(params, options))
        return false;

    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto input = fc_params.inputs[0];
    if (input.GetLayout() == DataLayout::bfyx &&
        (input.X().LogicalDimPadded() != 1 || input.Y().LogicalDimPadded() != 1 || input.Z().LogicalDimPadded() != 1)) {
        return false;
    }

    return true;
}

FullyConnectedKernelMMAD::DispatchData FullyConnectedKernelMMAD::SetDefault(const fully_connected_params& params,
                                                                            int) const {
    auto runInfo = Parent::SetDefault(params);

    const auto& out = params.output;

    std::vector<size_t> global = { Align(out.Feature().v, sub_group_size), out.Batch().v, 1 };
    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

JitConstants FullyConnectedKernelMMAD::GetJitConstants(const fully_connected_params& params,
                                                       const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    auto& input = params.inputs[0];
    auto& weights = params.weights;

    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    if (input.GetDims().size() == 5) {
        jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_YX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0)"));
    } else {
        jit.AddConstant(MakeJitConstant("FILTER_GET_OFFSET(f)", "GET_FILTER_OS_IS_ZYX_ISA8_OSV8_ISV4_INDEX(FILTER, f, 0, 0, 0, 0)"));
    }

    Datatype input_packed_type = Datatype::INT32;
    Datatype filter_packed_type = Datatype::INT32;

    if (input.GetDType() == Datatype::UINT8) {
        input_packed_type = Datatype::UINT32;
    } else if (input.GetDType() == Datatype::INT8) {
        input_packed_type = Datatype::INT32;
    }

    if (weights.GetDType() == WeightsType::UINT8) {
        filter_packed_type = Datatype::UINT32;
    } else if (weights.GetDType() == WeightsType::INT8) {
        filter_packed_type = Datatype::INT32;
    }

    jit.Merge(MakeTypeJitConstants(input_packed_type, "INPUT_PACKED"));
    jit.Merge(MakeTypeJitConstants(filter_packed_type, "FILTER_PACKED"));

    auto filter_spatial_size = weights.X().v * weights.Y().v * weights.Z().v;
    int filter_spatial_pitch = 4 * 8 * 8;

    jit.AddConstant(MakeJitConstant("FILTER_SPATIAL_SIZE", filter_spatial_size));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_SPATIAL_PITCH", filter_spatial_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_FILTER_FBLOCK_PITCH", filter_spatial_size * filter_spatial_pitch));

    size_t input_x_pitch = input.X().pitch;
    size_t input_y_pitch = input.Y().pitch;
    size_t input_z_pitch = input.Z().pitch;

    if (input.GetLayout() == DataLayout::byxf_af32 || input.GetLayout() == DataLayout::bfyx) {
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", 32));
    } else if (input.GetLayout() == DataLayout::b_fs_yx_fsv32 || input.GetLayout() == DataLayout::b_fs_zyx_fsv32) {
        input_x_pitch = 32;
        input_y_pitch *= 32;
        input_z_pitch *= 32;
        jit.AddConstant(MakeJitConstant("MMAD_INPUT_FBLOCK_PITCH", input.Feature().pitch * 32));
    }

    if (input.GetLayout() == DataLayout::bfyx && input.Feature().v % 32 != 0) {
        jit.AddConstant(MakeJitConstant("HAS_FEATURE_LEFTOVERS", true));
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCKS_COUNT", input.Feature().v / 32));
    } else {
        jit.AddConstant(MakeJitConstant("FEATURE_BLOCKS_COUNT", CeilDiv(input.Feature().v, 32)));
    }

    jit.AddConstant(MakeJitConstant("MMAD_INPUT_SPATIAL_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_X_PITCH", input_x_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Y_PITCH", input_y_pitch));
    jit.AddConstant(MakeJitConstant("MMAD_INPUT_Z_PITCH", input_z_pitch));

    bool split_spatial = input.X().pad.Total() != 0 || input.Y().pad.Total() != 0 || input.Z().pad.Total() != 0;
    bool spatial_major = DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::X) <
                         DataTensor::Channelndex(input.GetLayout(), Tensor::DataChannelName::FEATURE);

    jit.AddConstant(MakeJitConstant("SPLIT_SPATIAL", split_spatial));
    jit.AddConstant(MakeJitConstant("SPATIAL_MAJOR", spatial_major));

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf = { "", {"b", "f", "0", "0"}, "dequantized", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}

KernelsData FullyConnectedKernelMMAD::GetKernelsData(const Params& params, const optional_params& options) const {
    auto fc_params = static_cast<const fully_connected_params&>(params);
    auto& input = fc_params.inputs[0];

    auto w_layout = WeightsLayout::os_is_yx_isa8_osv8_isv4;
    if (input.GetDims().size() == 5) {
        w_layout = WeightsLayout::os_is_zyx_isa8_osv8_isv4;
    }

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    options,
                                                    input.GetLayout(),
                                                    w_layout,
                                                    FORCE_PRIORITY_9,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }
    return res;
}
}  // namespace kernel_selector
