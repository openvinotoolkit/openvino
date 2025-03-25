// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_bs_f_bsv16_b1.h"

namespace kernel_selector {
ParamsKey FullyConnected_bs_f_bsv16_b1::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::bf);
    k.EnableOutputLayout(DataLayout::bf);
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

DeviceFeaturesKey FullyConnected_bs_f_bsv16_b1::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();
    k.requires_subgroup_shuffle_relative();

    return k;
}

JitConstants FullyConnected_bs_f_bsv16_b1::GetJitConstants(
    const fully_connected_params& params,
    const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    auto& d = static_cast<const DispatchData&>(dispatchData);
    auto cldnn_jit = FullyConnectedKernelBase::GetJitConstants(params, dispatchData);
    cldnn_jit.AddConstants({
        MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[0]),
        MakeJitConstant("WORK_ITEMS_PER_BATCH", dispatchData.gws[1]),

        MakeJitConstant("UNIT_BYTE_SIZE", d.unit_byte_size),
        MakeJitConstant("CHUNK_TYPE", d.chunk_type),
        MakeJitConstant("CHUNK_BYTE_SIZE", d.chunk_byte_size),
        MakeJitConstant("UNITS_PER_CHUNK", d.units_per_chunk),
        MakeJitConstant("BYTES_PER_SG_READ", d.bytes_per_sg_read),
        MakeJitConstant("UNITS_PER_SG_READ", d.units_per_sg_read),
        MakeJitConstant("RESPONSES_PER_SG_EXEC", d.responses_per_sg_exec),
        MakeJitConstant("IN_CHUNK_PREFETCH_SIZE", d.in_chunk_prefetch_size),
        MakeJitConstant("FILTER_CHUNK_PREFETCH_SIZE", d.filter_chunk_prefetch_size),
    });
    return cldnn_jit;
}

FullyConnected_bs_f_bsv16_b1::DispatchData FullyConnected_bs_f_bsv16_b1::SetDefault(const fully_connected_params& arg,
                                                                                    int, int /*kernel_number*/) const {
    DispatchData dispatchData = FullyConnectedKernelBase::SetDefault(arg);

    // Properties of chunk and unit.
    const char* chunk_type = "uint";
    const uint32_t unit_byte_size = BytesPerElement(arg.inputs[0].GetDType());
    constexpr uint32_t chunk_byte_size = sizeof(uint32_t);
    constexpr uint32_t sub_group_size = 16;
    const uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
    const uint32_t units_per_sg_read = sub_group_size * units_per_chunk;
    // Properties of primitive responses.
    constexpr uint32_t responses_per_sg_exec =
        16;  // Must match batch slice size of weights format (bs_f_bsv16).
             // Number of response groups. Each group (except last) writes responses_per_sg_exec responses
             // for at least one input data set from batch.
    const auto response_size = arg.outputs[0].Feature().v;
    auto rg_count = CeilDiv(response_size, responses_per_sg_exec);

    dispatchData.lws[0] = sub_group_size;
    // Number of work items needed to process all response groups.
    dispatchData.gws[0] = rg_count * sub_group_size;
    dispatchData.lws[1] = dispatchData.lws[2] = 1;
    dispatchData.gws[1] = dispatchData.gws[2] = 1;

    dispatchData.unit_byte_size = unit_byte_size;
    dispatchData.chunk_type = chunk_type;
    dispatchData.chunk_byte_size = chunk_byte_size;
    dispatchData.units_per_chunk = units_per_chunk;
    dispatchData.bytes_per_sg_read = sub_group_size * chunk_byte_size;
    dispatchData.units_per_sg_read = units_per_sg_read;
    dispatchData.responses_per_sg_exec = responses_per_sg_exec;
    dispatchData.in_chunk_prefetch_size = 2;
    dispatchData.filter_chunk_prefetch_size = responses_per_sg_exec;

    return dispatchData;
}

bool FullyConnected_bs_f_bsv16_b1::Validate(const Params& p) const {
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

KernelsData FullyConnected_bs_f_bsv16_b1::GetKernelsData(const Params& params) const {
    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::bf,
                                                    WeightsLayout::os_i_osv16,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_bs_f_bsv16_b1::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
