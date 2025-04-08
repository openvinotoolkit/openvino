// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_yxfb_yxio_b16.h"
#include <string>
#include <algorithm>

namespace kernel_selector {

ParamsKey ConvolutionKernel_yxfb_yxio_b16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableGroupedConvolution();
    k.EnableDilation();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_yxfb_yxio_b16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

std::string ConvolutionKernel_yxfb_yxio_b16::GetKernelName(const convolution_params& params) const {
    if (params.inputs[0].GetDType() == Datatype::F32) {
        return kernelName + "_fp32";
    } else {
        return kernelName + "_fp16";
    }
}

namespace {
// how many batches will a single work item compute
size_t GetBatchesPerWorkItem(size_t batch_size, Datatype dataType) {
    if (dataType == Datatype::F16) {
        const uint32_t min_batches_per_wi = 1;
        const uint32_t min_lws = 16;

        if (batch_size % (4 * min_batches_per_wi * min_lws) == 0) {
            return 4 * min_batches_per_wi;  // USE_BLOCK_READ_2 + as_half4
        } else if (batch_size % (2 * min_batches_per_wi * min_lws) == 0) {
            return 2 * min_batches_per_wi;  // USE_BLOCK_READ_1 + as_half2
        } else {
            return min_batches_per_wi;
        }
    } else {
        return 2;
    }
}

size_t GetOfmPerWorkitem(Datatype dataType) {
    if (dataType == Datatype::F16)
        return 16;
    return 8;
}
}  // namespace

ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b16::SetDefault(const convolution_params& arg,
                                                                                int) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg);

    const auto filter_ofm_num = arg.weights.OFM().v * arg.weights.G().v;
    const auto batch_size = arg.outputs[0].Batch().v;
    const uint32_t min_lws = 16;

    const size_t batchesPerWorkItem = GetBatchesPerWorkItem(batch_size, arg.inputs[0].GetDType());
    const size_t ofmPerWorkItem = GetOfmPerWorkitem(arg.inputs[0].GetDType());

    dispatchData.lws[0] = min_lws;
    dispatchData.gws[0] = filter_ofm_num * batch_size / (ofmPerWorkItem * batchesPerWorkItem);

    if (arg.inputs[0].GetDType() == Datatype::F16) {
        dispatchData.lws[1] = 1;
        dispatchData.lws[2] = 1;
    }

    return dispatchData;
}

KernelsPriority ConvolutionKernel_yxfb_yxio_b16::GetKernelsPriority(const Params& params) const {
    const auto& p = static_cast<const convolution_params&>(params);

    return p.inputs[0].GetDType() == Datatype::F16 ? FORCE_PRIORITY_7 : FORCE_PRIORITY_9;
}

bool ConvolutionKernel_yxfb_yxio_b16::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }
    const convolution_params& params = static_cast<const convolution_params&>(p);

    const auto filter_ofm_num = params.weights.OFM().v;
    const auto filter_groups_num = params.weights.G().v;
    const auto batch_size = params.outputs[0].Batch().v;
    const uint32_t min_lws = 16;

    const bool bInputValidated =
        (filter_ofm_num > 0) && (batch_size > 0) && (params.outputs[0].Feature().v == filter_ofm_num * filter_groups_num);

    if (!bInputValidated) {
        return false;
    }

    if (params.inputs[0].GetDType() == Datatype::F16) {
        const uint32_t min_ofm_per_wi = 16;
        const uint32_t min_batches_per_wi = 1;

        const bool bFilterOK =
            filter_ofm_num % min_ofm_per_wi ==
            0;  // Number of output features dividable by minimum number of output features processed inside work item.
        const bool bBatchOK =
            batch_size % (min_batches_per_wi * min_lws) ==
            0;  // Batch size dividable by minimum number of batches processed when smallest local work size is used.

        if (!bFilterOK || !bBatchOK) {
            return false;
        }
    } else {
        if ((filter_ofm_num * batch_size) % min_lws != 0 || batch_size < 32) {  // TODO: check why it's not supported
            return false;
        }
    }

    return true;
}

JitConstants ConvolutionKernel_yxfb_yxio_b16::GetJitConstants(const convolution_params& params,
                                                              const DispatchData& dispatchData) const {
    auto jit = Parent::GetJitConstants(params, dispatchData);

    const auto local_work_group_size = dispatchData.lws[0];
    const auto batch_size = params.outputs[0].Batch().v;

    if (params.inputs[0].GetDType() == Datatype::F32) {
        // A LITTLE HACK, for convolutions with low number of input features don't use block reads, and it will speed up
        // by 25%
        // TODO - investigate why is this happening
        if (params.inputs[0].Feature().v > 4) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
        }
    } else {
        const auto batch_pad_before = params.outputs[0].Batch().pad.before;
        const auto feature_pitch = params.outputs[0].Feature().pitch;

        if (batch_size >= 64 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0)) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_2", ""));
        } else if (batch_size >= 32 && (feature_pitch % 2 == 0) && (batch_pad_before % 2 == 0)) {
            jit.AddConstant(MakeJitConstant("USE_BLOCK_READ_1", ""));
        }
    }

    const size_t batchesPerWorkItem = GetBatchesPerWorkItem(batch_size, params.inputs[0].GetDType());
    const size_t ofmPerWorkItem = GetOfmPerWorkitem(params.inputs[0].GetDType());

    jit.AddConstants({
        MakeJitConstant("LOCAL_WORK_GROUP_SIZE", dispatchData.lws[0]),
        MakeJitConstant("OFM_PER_WORK_ITEM", ofmPerWorkItem),
        MakeJitConstant("BATCHES_PER_WORK_ITEM",
                        batchesPerWorkItem),  // how many batches will a single work item compute
        MakeJitConstant(
            "LOCAL_WORK_GROUPS_PER_SINGLE_BATCHES_ELEMENTS",
            std::max(batch_size / batchesPerWorkItem / local_work_group_size,
                     static_cast<size_t>(
                         1))),  // how many local work groups we need to compute single element for each batch
        MakeJitConstant(
            "WORK_ITEMS_PER_SINGLE_BATCHES_ELEMENTS",
            batch_size / batchesPerWorkItem),  // how many work items we need to compute single element for each batch
    });

    return jit;
}

KernelsData ConvolutionKernel_yxfb_yxio_b16::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
