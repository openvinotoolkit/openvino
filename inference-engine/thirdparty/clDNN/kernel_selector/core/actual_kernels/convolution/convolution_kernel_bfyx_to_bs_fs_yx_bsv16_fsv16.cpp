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


#include "convolution_kernel_bfyx_to_bs_fs_yx_bsv16_fsv16.h"
#include "convolution_kernel_bfyx_to_b_fs_yx_fsv16.h"
#include "kernel_selector_utils.h"
#include <vector>
#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
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
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::SetDefault(const convolution_params& params,
                                                                                   int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernel_bfyx_to_bfyx_f16::SetDefault(params, autoTuneIndex);

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}

bool ConvolutionKernel_bfyx_to_bfyx_bsv16_fsv16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

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
