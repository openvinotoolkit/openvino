//
// Copyright (c) 2019 Intel Corporation
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

#include "deconvolution_kernel_bfzyx_f16.h"
#include "kernel_selector_utils.h"

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ParamsKey DeconvolutionKernel_bfzyx_f16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::bfzyx_f16);
    k.EnableOutputLayout(DataLayout::bfzyx_f16);
    k.EnableInputLayout(DataLayout::bfzyx_b16f16);
    k.EnableOutputLayout(DataLayout::bfzyx_b16f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

DeconvolutionKernelBase::DispatchData DeconvolutionKernel_bfzyx_f16::SetDefault(const deconvolution_params& params) const {
    DispatchData kd = DeconvolutionKernelBase::SetDefault(params);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    kd.gws0 = f;
    kd.gws1 = x * y * z;
    kd.gws2 = (b % 16 == 0)? b / 16 : CeilDiv(b, 16);

    kd.lws0 = sub_group_size;
    kd.lws1 = 1;
    kd.lws2 = 1;

    if (b == 1)
        kd.effiency = FORCE_PRIORITY_2;
    else
        kd.effiency = FORCE_PRIORITY_7;

    return kd;
}

bool DeconvolutionKernel_bfzyx_f16::Validate(const Params& p, const optional_params& o) const {
    if (!DeconvolutionKernelBase::Validate(p, o)) {
        return false;
    }

    const auto& params = static_cast<const deconvolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    if (output.Feature().v % feature_block_size != 0)
        return false;

    if (input.Feature().v % feature_block_size != 0)
        return false;

    // Check that padding before features doesn't miss-align the blocks
    if (input.Feature().pad.before % feature_block_size != 0 || output.Feature().pad.before % feature_block_size != 0) {
        return false;
    }

    return true;
}

JitConstants DeconvolutionKernel_bfzyx_f16::GetJitConstants(const deconvolution_params& params) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params);

    if (output.Batch().v % 16 == 0) {
        jit.AddConstant(MakeJitConstant("VER_16MB16C", 1));
    } else {
        jit.AddConstant(MakeJitConstant("VER_8OW16C", 1));
    }
    jit.AddConstant(MakeJitConstant("OC_BLOCK", 16));
    jit.AddConstant(MakeJitConstant("NCHW", 1));
    jit.AddConstant(MakeJitConstant("CASE_3D", 1));

    if (output.GetDType() == Datatype::F32)
        jit.AddConstant(MakeJitConstant("DT_F32", 1));
    else
        jit.AddConstant(MakeJitConstant("DT_F16", 1));

    // the conditional code below was replaced to fix security issue
    // auto is_1stconv = false;
    // auto mb_block =(is_1stconv && output.Batch().v % 16 == 0) ? 16 : 1;
    // auto ic_block = (is_1stconv) ? 1 : 16;
    auto mb_block = 1;
    auto ic_block = 16;

    if (output.Batch().v % 16 == 0) {
        jit.AddConstant(MakeJitConstant("MB_BLOCK", 16));
        jit.AddConstant(MakeJitConstant("IC_BLOCK", 16));
    } else {
        jit.AddConstant(MakeJitConstant("MB_BLOCK", mb_block));
        jit.AddConstant(MakeJitConstant("IC_BLOCK", ic_block));
    }
    jit.AddConstant(MakeJitConstant("MB_LAST", (output.Batch().v / 16) * 16));
    jit.AddConstant(MakeJitConstant("G", params.split));
    jit.AddConstant(MakeJitConstant("DD", params.dilation.z - 1));
    jit.AddConstant(MakeJitConstant("DH", params.dilation.y - 1));
    jit.AddConstant(MakeJitConstant("DW", params.dilation.x - 1));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("IS_DW", "DEPTHWISE_SEPARABLE_OPT"));
    jit.AddConstant(MakeJitConstant("BWD_DATA", 1));
    jit.AddConstant(MakeJitConstant("WITH_BIAS", "BIAS_TERM"));

    jit.AddConstant(MakeJitConstant("MB", "OUTPUT_BATCH_NUM"));
    jit.AddConstant(MakeJitConstant("OC", "INPUT0_FEATURE_NUM"));
    jit.AddConstant(MakeJitConstant("OD", "INPUT0_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("OH", "INPUT0_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("OW", "INPUT0_SIZE_X"));
    jit.AddConstant(MakeJitConstant("IC", "OUTPUT_FEATURE_NUM"));
    jit.AddConstant(MakeJitConstant("ID", "OUTPUT_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("IH", "OUTPUT_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("IW", "OUTPUT_SIZE_X"));
    jit.AddConstant(MakeJitConstant("KD", "FILTER_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("KH", "FILTER_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("KW", "FILTER_SIZE_X"));
    jit.AddConstant(MakeJitConstant("SD", "STRIDE_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("SH", "STRIDE_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("SW", "STRIDE_SIZE_X"));
    jit.AddConstant(MakeJitConstant("PD", "PADDING_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("PH", "PADDING_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("PW", "PADDING_SIZE_X"));

    return jit;
}

}  // namespace kernel_selector
