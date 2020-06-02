//
// Copyright (c) 2020 Intel Corporation
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
//

#include "deconvolution_kernel_b_fs_zyx_fsv16_dw.h"
#include "kernel_selector_utils.h"

#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

size_t DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetBlockSizeX(const deconvolution_params& params) const {
    std::vector<size_t> blockWidths = {8, 4, 2, 1};
    for (auto& blockSize : blockWidths)
        if (params.output.X().v % blockSize == 0) {
            return blockSize;
        }
    return 1;
}

ParamsKey DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    k.EnableGroupedConvolution();
    return k;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_b_fs_zyx_fsv16_dw::SetDefault(const deconvolution_params& params) const {
    DispatchData kd = DeconvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = (x / GetBlockSizeX(params)) * y * z;
    kd.gws1 = Align(f, feature_block_size);
    kd.gws2 = b;

    kd.lws0 = 1;
    kd.lws1 = sub_group_size;
    kd.lws2 = 1;

    kd.efficiency = FORCE_PRIORITY_2;

    return kd;
}

bool DeconvolutionKernel_b_fs_zyx_fsv16_dw::Validate(const Params& p, const optional_params& o) const {
    if (!DeconvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const deconvolution_params&>(p);

    if (params.groups == 1)
        return false;

    if (params.weights.IFM().v != 1 || params.weights.OFM().v != 1)
        return false;

    // Check that padding features doesn't miss-align the blocks
    if (params.inputs[0].Feature().pad.before % feature_block_size != 0 || params.output.Feature().pad.before % feature_block_size != 0)
        return false;

    return true;
}

JitConstants DeconvolutionKernel_b_fs_zyx_fsv16_dw::GetJitConstants(const deconvolution_params& params) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params);

    jit.AddConstant(MakeJitConstant("X_BLOCK_SIZE", GetBlockSizeX(params)));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    if (params.output.Feature().v % feature_block_size != 0) {
        jit.AddConstant(MakeJitConstant("OUTPUT_LEFTOVERS", params.output.Feature().v % feature_block_size));
    }

    return jit;
}

}  // namespace kernel_selector
