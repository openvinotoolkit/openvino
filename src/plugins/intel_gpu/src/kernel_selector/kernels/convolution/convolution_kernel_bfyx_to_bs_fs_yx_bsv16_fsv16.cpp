// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution_kernel_bfyx_to_bs_fs_yx_bsv16_fsv16.h"
#include "convolution_kernel_bfyx_to_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {

static const size_t feature_block_size = 16;
static const size_t batch_block_size = 16;

ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16()
    : ConvolutionKernel_bfyx_to_bfyx_f16("convolution_gpu_bfyx_to_bs_fs_yx_bsv16_fsv16") {}

ParamsKey ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bs_fs_yx_bsv16_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    return k;
}

DeviceFeaturesKey ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::get_required_device_features_key(const Params& params) const {
    auto k = get_common_subgroups_device_features_key(params);
    k.requires_subgroup_shuffle();

    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::SetDefault(const convolution_params& params,
                                                                                           int autoTuneIndex) const {
    DispatchData dispatchData = ConvolutionKernel_bfyx_to_bfyx_f16::SetDefault(params, autoTuneIndex);

    return dispatchData;
}

KernelsPriority ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::GetKernelsPriority(const Params& /*params*/) const {
    return FORCE_PRIORITY_2;
}

bool ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::Validate(const Params& p) const {
    if (!ConvolutionKernelBase::Validate(p)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.outputs[0];

    if (output.Feature().v % feature_block_size != 0 || output.Batch().v % batch_block_size != 0)
        return false;

    if (input.Feature().v != 3) {
        return false;
    }

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    return true;
}

}  // namespace kernel_selector
