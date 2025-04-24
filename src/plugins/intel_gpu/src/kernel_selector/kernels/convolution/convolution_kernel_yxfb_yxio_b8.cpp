// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_yxfb_yxio_b8.h"

namespace kernel_selector {

ParamsKey ConvolutionKernel_yxfb_yxio_b8::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableDilation();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_yxfb_yxio_b8::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

namespace {
size_t GetOfmPerWorkitem(size_t filterOfmNum, size_t batchSize, size_t local_work_size) {
    if (((filterOfmNum * batchSize) / 16) % local_work_size) {
        return 8;
    } else {
        return 16;
    }
}
}  // namespace

ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b8::SetDefault(const convolution_params& arg,
                                                                               int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg, autoTuneIndex);

    const auto filterOfmNum = arg.weights.OFM().v;
    const auto batchSize = arg.outputs[0].Batch().v;

    dispatchData.lws[0] = batchSize == 8 ? 8 : 16;
    dispatchData.lws[1] = 1;
    dispatchData.lws[2] = 1;

    size_t ofmPerWorkItem = GetOfmPerWorkitem(filterOfmNum, batchSize, dispatchData.lws[0]);

    dispatchData.gws[0] = filterOfmNum * batchSize / ofmPerWorkItem;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_yxfb_yxio_b8::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_9;
}

bool ConvolutionKernel_yxfb_yxio_b8::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);
    const auto filterOfmNum = params.weights.OFM().v;
    const auto batchSize = params.outputs[0].Batch().v;

    const bool bInputValidated = (filterOfmNum > 0) && (batchSize > 0) && (params.outputs[0].Feature().v == filterOfmNum);

    if (!bInputValidated) {
        return false;
    }

    const uint32_t lws0 = batchSize == 8 ? 8 : 16;

    if ((filterOfmNum * batchSize) % lws0 != 0 || batchSize > 16 || batchSize == 1) {
        return false;
    }

    if (params.outputs[0].PitchesDifferFromLogicalDims())
        return false;

    return true;
}

JitConstants ConvolutionKernel_yxfb_yxio_b8::GetJitConstants(const convolution_params& params,
                                                             const DispatchData& dispatchData) const {
    JitConstants jits = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    size_t ofmPerWorkItem = GetOfmPerWorkitem(params.weights.OFM().v, params.outputs[0].Batch().v, dispatchData.lws[0]);

    jits.AddConstant(MakeJitConstant("OFM_PER_WORK_ITEM", ofmPerWorkItem));
    jits.AddConstant(MakeJitConstant("LOCAL_WORK_GROUP_SIZE", dispatchData.lws[0]));

    return jits;
}

KernelsData ConvolutionKernel_yxfb_yxio_b8::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
