// Copyright (c) 2016-2019 Intel Corporation
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

#include "deconvolution_kernel_ref.h"

namespace kernel_selector {

ParamsKey DeconvolutionKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);

    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputWeightsType(WeightsType::UINT8);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);

    k.EnableInputLayout(DataLayout::yxfb);
    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableInputLayout(DataLayout::byxf);
    k.EnableInputLayout(DataLayout::bfzyx);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableInputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);
    k.EnableOutputLayout(DataLayout::yxfb);
    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::byxf);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableOutputLayout(DataLayout::b_fs_zyx_fsv16);
    k.EnableOutputLayout(DataLayout::bs_fs_zyx_bsv16_fsv16);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSplitSupport();
    k.EnableDepthwiseSeparableOpt();
    k.EnableGroupedConvolution();
    k.EnableDifferentTypes();
    k.EnableDifferentInputWeightsTypes();
    return k;
}

CommonDispatchData DeconvolutionKernelRef::SetDefault(const deconvolution_params& params) const {
    CommonDispatchData runInfo = DeconvolutionKernelBase::SetDefault(params);

    if (params.output.Feature().v * params.output.Batch().v <= 16) {
        const auto& out = params.output;
        runInfo.gws0 = Align(out.X().v, 32);
        runInfo.gws1 = out.Y().v * out.Z().v;
        runInfo.gws2 = out.Feature().v * out.Batch().v;

        runInfo.lws0 = 32;
        runInfo.lws1 = 1;
        runInfo.lws2 = 1;
    }

    return runInfo;
}

JitConstants DeconvolutionKernelRef::GetJitConstants(const deconvolution_params& params) const {
    auto jit = DeconvolutionKernelBase::GetJitConstants(params);

    if (params.output.Feature().v * params.output.Batch().v <= 16)
        jit.AddConstant(MakeJitConstant("DIM_ORDER_XYBF", 1));

    if (!params.fused_ops.empty()) {
        auto fused_dt = GetActivationType(params);
        std::vector<std::string> idx_order;
        if (params.output.Dimentions() <= 4) {
            idx_order = { "batch_offset", "ofm_offset", "out_y", "out_x" };
        } else {
            idx_order = { "batch_offset", "ofm_offset", "out_z", "out_y", "out_x" };
        }
        FusedOpsConfiguration conf = { "", idx_order, "post_activation", fused_dt, 1, LoadType::LT_UNALIGNED, BoundaryCheck::DISABLED };

        jit.Merge(MakeFusedOpsJitConstants(params, { conf }));
    }

    return jit;
}
}  // namespace kernel_selector
