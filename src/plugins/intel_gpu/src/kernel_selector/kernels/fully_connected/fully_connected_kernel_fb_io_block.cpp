// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fully_connected_kernel_fb_io_block.h"

namespace kernel_selector {
ParamsKey FullyConnected_fb_io_block::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableAllInputLayout();
    k.EnableOutputLayout(DataLayout::fb);
    k.EnableBatching();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    return k;
}

DeviceFeaturesKey FullyConnected_fb_io_block::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

FullyConnected_fb_io_block::DispatchData FullyConnected_fb_io_block::SetDefault(const fully_connected_params& arg,
                                                                                int, int /*kernel_number*/) const {
    auto dispatchData = FullyConnectedKernelBase::SetDefault(arg);
    const auto& output = arg.outputs[0];

    auto batch_size = output.Batch().v;
    auto response_size = output.Feature().v;

    constexpr uint32_t unit_byte_size = sizeof(short);
    const char* chunk_type = "uint";
    constexpr uint32_t chunk_byte_size = sizeof(uint32_t);
    constexpr uint32_t sub_group_size = 16;
    constexpr uint32_t units_per_chunk = chunk_byte_size / unit_byte_size;
    constexpr uint32_t units_per_sg_read = sub_group_size * units_per_chunk;

    // Number of response groups. Each group (except last) writes units_per_sg_read responses
    // for at least one input data set from batch.
    auto rg_count = CeilDiv(response_size, units_per_sg_read);

    dispatchData.lws[0] = sub_group_size;
    // Number of work items needed to process all response groups.
    dispatchData.gws[0] = rg_count * sub_group_size;
    dispatchData.lws[1] = 1;
    dispatchData.gws[1] = batch_size / units_per_sg_read;

    dispatchData.unit_byte_size = unit_byte_size;
    dispatchData.chunk_type = chunk_type;
    dispatchData.chunk_byte_size = chunk_byte_size;
    dispatchData.units_per_chunk = units_per_chunk;
    dispatchData.bytes_per_sg_read = sub_group_size * chunk_byte_size;
    dispatchData.units_per_sg_read = units_per_sg_read;
    dispatchData.rg_count = (uint32_t)rg_count;
    dispatchData.last_rg_size = response_size % units_per_sg_read;
    return dispatchData;
}

JitConstants FullyConnected_fb_io_block::GetJitConstants(const fully_connected_params& params,
                                                         const FullyConnectedKernelBase::DispatchData& dispatchData) const {
    auto cldnn_jit = FullyConnectedKernelBase::GetJitConstants(params, dispatchData);
    cldnn_jit.AddConstants({
        MakeJitConstant("SUB_GROUP_SIZE", dispatchData.lws[0]),
        MakeJitConstant("WORK_ITEMS_PER_BATCH", dispatchData.gws[1]),
        MakeJitConstant("UNIT_BYTE_SIZE", dispatchData.unit_byte_size),
        MakeJitConstant("CHUNK_TYPE", dispatchData.chunk_type),
        MakeJitConstant("CHUNK_BYTE_SIZE", dispatchData.chunk_byte_size),
        MakeJitConstant("UNITS_PER_CHUNK", dispatchData.units_per_chunk),
        MakeJitConstant("BYTES_PER_SG_READ", dispatchData.bytes_per_sg_read),
        MakeJitConstant("UNITS_PER_SG_READ", dispatchData.units_per_sg_read),
        MakeJitConstant("RG_COUNT", dispatchData.rg_count),
        MakeJitConstant("LAST_RG_SIZE", dispatchData.last_rg_size),
    });
    return cldnn_jit;
}

bool FullyConnected_fb_io_block::Validate(const Params& p) const {
    if (!FullyConnectedKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const fully_connected_params&>(p);

    const auto& output = params.outputs[0];
    const auto responseSize = output.Feature().v;
    const auto batches = output.Batch().v;
    const auto xSize = output.LogicalSize() / batches;

    constexpr uint32_t subGroupSize = 16;
    constexpr uint32_t bytesPerElement = sizeof(short);
    constexpr uint32_t chunkSizeInBytes = sizeof(uint32_t);
    constexpr uint32_t chunkSizeInElements = chunkSizeInBytes / bytesPerElement;
    constexpr uint32_t elementsPerBlockRead = subGroupSize * chunkSizeInElements;

    const bool bSupportedBatch = (batches > 0) && ((batches % 8) == 0) && ((batches % elementsPerBlockRead) == 0);

    const bool bSupportedFeature =
        (responseSize > 0) && (((responseSize * bytesPerElement) % 4) == 0) && ((xSize % 8) == 0);

    if (!bSupportedBatch || !bSupportedFeature) {
        return false;
    }

    if (!params.bias.empty()) {
        if (params.inputs[0].GetDType() != params.bias[0].GetDType()) {
            return false;
        }
    }

    return true;
}

KernelsData FullyConnected_fb_io_block::GetKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::FULLY_CONNECTED);

    // TODO: it should be fb_io. but the original code use this kernel with yxfb and yxio
    //       (fb == fyxb flatten fyx, not yxfb flatten yxf).
    //       the order of the add operation cause some numeric changes. in order to avoid them right now we use
    //       yxfb/oiyx instead.
    // return GetCommonKernelsData(params,  DataLayout::fb, WeightsLayout::io, estimated_time);
    // return GetCommonKernelsData(params,  DataLayout::yxfb, { WeightsLayout::yxio }, estimated_time);

    KernelsData res = {};
    for (size_t i = 0; i < autoTuneOptions.size(); i++) {
        KernelsData kd = GetTunedKernelsDataByIndex(params,
                                                    DataLayout::yxfb,
                                                    WeightsLayout::yxio,
                                                    static_cast<int>(i));
        if (!kd.empty()) {
            res.emplace_back(kd[0]);
        }
    }

    return res;
}

KernelsPriority FullyConnected_fb_io_block::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const fully_connected_params&>(params);

    return p.inputs[0].GetDType() == Datatype::F16 && p.outputs[0].Batch().v >= 16 ? FORCE_PRIORITY_3
                                                                               : FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
