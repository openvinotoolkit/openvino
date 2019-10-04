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

#include "convolution_kernel_bfzyx_f16.h"
#include "kernel_selector_utils.h"
#include <algorithm>

namespace kernel_selector {

static const size_t sub_group_size = 16;
static const size_t feature_block_size = 16;

ParamsKey ConvolutionKernel_bfzyx_f16::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);
    k.EnableInputWeightsType(WeightsType::F32);
    k.EnableInputWeightsType(WeightsType::F16);
    k.EnableInputLayout(DataLayout::bfzyx_f16);
    k.EnableOutputLayout(DataLayout::bfzyx_f16);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableNonBiasTerm();
    k.EnableSplitSupport();
    k.EnableBatching();
    k.EnableSubGroup();
    k.EnableSubGroupShort();
    return k;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_bfzyx_f16::SetDefault(const convolution_params& params,
                                                                           int autoTuneIndex) const {
    DispatchData kd = ConvolutionKernelBase::SetDefault(params, autoTuneIndex);

    const auto& out = params.output;

    auto x = out.X().v;
    auto y = out.Y().v;
    auto z = out.Z().v;
    auto f = out.Feature().v;
    auto b = out.Batch().v;

    auto oh_block = 1;

    auto div = 16;
    while (div > 1) {
        if (x % div == 0)
            break;
        div--;
    }
    auto ow_block = std::max(8, div);

    auto ocb = 128;
    while (ocb > 16) {
        if (f % ocb == 0)
            break;
        else
            ocb /= 2;
    }

    kd.cldnnStyle.blockWidth = ow_block;

    kd.gws0 = ocb;
    kd.gws1 = CeilDiv(y, oh_block) * CeilDiv(x, ow_block) * z;
    kd.gws2 = b * (f / ocb);

    kd.lws0 = sub_group_size;
    kd.lws1 = 1;
    kd.lws2 = 1;

    if (b == 1)
        kd.effiency = FORCE_PRIORITY_2;
    else
        kd.effiency = FORCE_PRIORITY_7;

    return kd;
}

bool ConvolutionKernel_bfzyx_f16::Validate(const Params& p, const optional_params& o) const {
    if (!ConvolutionKernelBase::Validate(p, o) || !CovolutionCheckInput(p, o)) {
        return false;
    }

    const auto& params = static_cast<const convolution_params&>(p);

    const auto& input = params.inputs[0];
    const auto& output = params.output;

    if (output.GetDType() != use_data_type)
        return false;

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

JitConstants ConvolutionKernel_bfzyx_f16::GetJitConstants(const convolution_params& params,
                                                         const DispatchData& runInfo) const {
    auto input = params.inputs[0];
    auto output = params.output;
    auto jit = Parent::GetJitConstants(params, runInfo);

    jit.AddConstant(MakeJitConstant("VER_8OW16C", 1));
    jit.AddConstant(MakeJitConstant("OC_BLOCK", 16));
    jit.AddConstant(MakeJitConstant("NCHW", 1));
    jit.AddConstant(MakeJitConstant("CASE_3D", 1));

    jit.AddConstant(MakeJitConstant("LWS_0", runInfo.lws0));
    jit.AddConstant(MakeJitConstant("LWS_1", runInfo.lws1));
    jit.AddConstant(MakeJitConstant("LWS_2", runInfo.lws2));

    jit.AddConstant(MakeJitConstant("OCB", runInfo.gws0));

    jit.AddConstant(MakeJitConstant("SUM_SCALE", 1));

    auto blockWidth = runInfo.cldnnStyle.blockWidth;
    // the conditional code below was replaced to fix security issue
    // auto is_1stconv = false;
    // auto mb_block =(is_1stconv && output.Batch().v % 16 == 0) ? 16 : 1;
    // auto ic_block = (is_1stconv) ? 1 : 16;
    auto mb_block = 1;
    auto ic_block = 16;

    jit.AddConstant(MakeJitConstant("MB_BLOCK", mb_block));
    jit.AddConstant(MakeJitConstant("MB_LAST", (output.Batch().v / 16) * 16));
    jit.AddConstant(MakeJitConstant("IC_BLOCK", ic_block));
    jit.AddConstant(MakeJitConstant("OH_BLOCK", 1));
    jit.AddConstant(MakeJitConstant("OW_BLOCK", blockWidth));
    jit.AddConstant(MakeJitConstant("OW_LAST", (output.X().v / blockWidth) * blockWidth));
    jit.AddConstant(MakeJitConstant("OWB", CeilDiv(output.X().v, blockWidth)));
    jit.AddConstant(MakeJitConstant("OHB", CeilDiv(output.Y().v, 1)));
    jit.AddConstant(MakeJitConstant("G", params.split));
    jit.AddConstant(MakeJitConstant("DD", params.dilation.z - 1));
    jit.AddConstant(MakeJitConstant("DH", params.dilation.y - 1));
    jit.AddConstant(MakeJitConstant("DW", params.dilation.x - 1));
    jit.AddConstant(MakeJitConstant("SUB_GROUP_SIZE", sub_group_size));
    jit.AddConstant(MakeJitConstant("FWD_DATA", 1));
    jit.AddConstant(MakeJitConstant("IS_DW", "DEPTHWISE_SEPARABLE_OPT"));
    jit.AddConstant(MakeJitConstant("WITH_BIAS", "BIAS_TERM"));

    jit.AddConstant(MakeJitConstant("MB", "OUTPUT_BATCH_NUM"));
    jit.AddConstant(MakeJitConstant("OC", "OUTPUT_FEATURE_NUM"));
    jit.AddConstant(MakeJitConstant("OD", "OUTPUT_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("OH", "OUTPUT_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("OW", "OUTPUT_SIZE_X"));
    jit.AddConstant(MakeJitConstant("IC", "INPUT0_FEATURE_NUM"));
    jit.AddConstant(MakeJitConstant("ID", "INPUT0_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("IH", "INPUT0_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("IW", "INPUT0_SIZE_X"));
    jit.AddConstant(MakeJitConstant("KD", "FILTER_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("KH", "FILTER_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("KW", "(FILTER_SIZE_X)"));
    jit.AddConstant(MakeJitConstant("SD", "STRIDE_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("SH", "STRIDE_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("SW", "STRIDE_SIZE_X"));
    jit.AddConstant(MakeJitConstant("PD", "PADDING_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("PH", "PADDING_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("PW", "PADDING_SIZE_X"));
    jit.AddConstant(MakeJitConstant("PD_R", "PADDING_SIZE_Z"));
    jit.AddConstant(MakeJitConstant("PH_R", "PADDING_SIZE_Y"));
    jit.AddConstant(MakeJitConstant("PW_R", "PADDING_SIZE_X"));

    return jit;
}

KernelsData ConvolutionKernel_bfzyx_f16::GetKernelsData(const Params& params, const optional_params& options) const {
    return GetTunedKernelsDataByIndex(params, options);
}
}  // namespace kernel_selector
