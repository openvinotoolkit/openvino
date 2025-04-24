// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_yxfb_yxio_b1_block_multiple_x.h"

namespace kernel_selector {

constexpr size_t local_work_size = 16;

ParamsKey ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::GetSupportedKey() const {
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

DeviceFeaturesKey ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::get_required_device_features_key(const Params& params) const {
    return get_common_subgroups_device_features_key(params);
}

namespace {
size_t GetOfmPerWorkitem(size_t filter_ofm_num, size_t localWorkSize) {
    if (filter_ofm_num % (localWorkSize * 4) == 0)
        return 4;
    if (filter_ofm_num % (localWorkSize * 2) == 0)
        return 2;
    return 1;
}
}  // namespace

ConvolutionKernelBase::DispatchData ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::SetDefault(
    const convolution_params& arg,
    int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernelBase::SetDefault(arg, autoTuneIndex);

    const auto filter_ofm_num = arg.weights.OFM().v;
    const auto batch_size = arg.outputs[0].Batch().v;

    dispatchData.lws[0] = local_work_size;

    // We cannot return 8 because we are processing 4 spatial coordinates for batch1,
    // and if we use more than 4 ofm_per_work_item we downgrade simd16 to simd8 which would break this algorithm.
    // NOTE: We could return 8 but then we must process only 2 coordinates, which is slower than processing 4
    // coordinates using blockread4
    // TODO: experiment with SIMD8 version of algorithm and check if it could be faster
    /*if (output_feature_count % (lws * 8) == 0)
        {
        dispatchData.ofm_per_work_item = 8;
        dispatchData.gws[1] = static_cast<size_t>(std::ceil(static_cast<float>(dispatchData.gws[1]) / 2.0f));
        }
        else*/
    const size_t ofmPerWorkItem = GetOfmPerWorkitem(filter_ofm_num, local_work_size);
    if (ofmPerWorkItem == 4) {
        // We compute multiple spatial coordinates "x" in a single workitem that's why we must divide
        dispatchData.gws[1] = static_cast<size_t>(std::ceil(static_cast<float>(dispatchData.gws[1]) / 4.0f));
    } else if (ofmPerWorkItem == 2) {
        dispatchData.gws[1] = static_cast<size_t>(std::ceil(static_cast<float>(dispatchData.gws[1]) / 8.0f));
    } else {
        dispatchData.gws[1] = static_cast<size_t>(std::ceil(static_cast<float>(dispatchData.gws[1]) / 8.0f));
    }

    dispatchData.gws[0] = filter_ofm_num * batch_size / ofmPerWorkItem;

    return dispatchData;
}

KernelsPriority ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::GetKernelsPriority(const Params& /*params*/) const {
    return DONT_USE_IF_HAVE_SOMETHING_ELSE;
}

JitConstants ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::GetJitConstants(const convolution_params& params,
                                                                              const DispatchData& dispatchData) const {
    auto cldnn_jit = ConvolutionKernelBase::GetJitConstants(params, dispatchData);

    size_t ofmPerWorkItem = GetOfmPerWorkitem(params.weights.OFM().v, local_work_size);
    cldnn_jit.AddConstant(MakeJitConstant("USE_VECTOR", ofmPerWorkItem));
    if (ofmPerWorkItem == 8) {
        cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 2));
    } else if (ofmPerWorkItem == 4) {
        cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 4));
    } else {
        cldnn_jit.AddConstant(MakeJitConstant("X_PER_WORK_ITEM", 8));
    }

    cldnn_jit.AddConstant(MakeJitConstant(
        "OFM_PER_WORK_ITEM",
        ofmPerWorkItem));  // how many output feature maps for a single batch will a single work item produce
    cldnn_jit.AddConstant(MakeJitConstant("LOCAL_WORK_GROUP_SIZE", dispatchData.lws[0]));
    return cldnn_jit;
}

bool ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const convolution_params& params = static_cast<const convolution_params&>(p);

    const auto filter_ofm_num = params.weights.OFM().v;
    const auto batch_size = params.outputs[0].Batch().v;

    const bool bInputValidated = (filter_ofm_num > 0) &&
                                 (batch_size == 1) &&  // current implementation doesn't support batching
                                                       // (subgorup is along batch*ofm and trying to block read
                                                       // filter/bias along batch and filter doesn't contain batching).
                                 (params.outputs[0].Feature().v == filter_ofm_num);

    if (!bInputValidated) {
        return false;
    }

    if ((filter_ofm_num * batch_size) % 16 != 0) {
        return false;
    }

    return true;
}

KernelsData ConvolutionKernel_yxfb_yxio_b1_block_multiple_x::GetKernelsData(const Params& params) const {
    return GetTunedKernelsDataByIndex(params);
}
}  // namespace kernel_selector
