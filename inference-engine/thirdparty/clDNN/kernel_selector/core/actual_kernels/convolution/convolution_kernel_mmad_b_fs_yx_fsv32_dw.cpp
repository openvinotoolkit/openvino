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

#include "convolution_kernel_mmad_b_fs_yx_fsv32_dw.h"
#include <vector>
#include <utility>
#include <string>
#include <algorithm>
#include <iostream>
#include <core/common/kernel_selector_utils.h>

namespace kernel_selector {

ParamsKey ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableInputWeightsType(WeightsType::INT8);
    k.EnableInputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableOutputLayout(DataLayout::b_fs_yx_fsv32);
    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBiasPerFeature();
    k.EnableBiasPerOutput();
    k.EnableNonBiasTerm();
    k.EnableBatching();
    k.EnableQuantization(QuantizationType::SYMMETRIC);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_DATA);
    k.EnableQuantization(QuantizationType::ASYMMETRIC_WEIGHTS);
    k.EnableDifferentTypes();
    k.DisableTuning();
    k.EnableGroupedConvolution();
    k.EnableDepthwiseSeparableOpt();
    k.EnableDifferentInputWeightsTypes();
    return k;
}


bool ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw::Validate(const Params& p, const optional_params& o) const {
    if (!Parent::Validate(p, o)) {
        return false;
    }

    auto params = dynamic_cast<const convolution_params&>(p);

    if (!params.depthwise_separable_opt)
        return false;

    if ((params.quantization == QuantizationType::ASYMMETRIC_DATA || params.quantization == QuantizationType::ASYMMETRIC_DATA_AND_WEIGHTS)
        && !params.has_compensation) {
        return false;
    }

    return true;
}

ConvolutionKernelBase::DispatchData ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw::SetDefault(const convolution_params& cp,
                                                                                        int /*autoTuneIndex*/) const {
    DispatchData runInfo = ConvolutionKernelBase::SetDefault(cp);

    runInfo.effiency = FORCE_PRIORITY_3;

    std::vector<size_t> global = {cp.output.Feature().v, cp.output.X().v * cp.output.Y().v, cp.output.Batch().v};

    runInfo.gws0 = global[0];
    runInfo.gws1 = global[1];
    runInfo.gws2 = global[2];

    auto local = GetOptimalLocalWorkGroupSizes(global, cp.engineInfo);
    runInfo.lws0 = local[0];
    runInfo.lws1 = local[1];
    runInfo.lws2 = local[2];

    return runInfo;
}

// TODO: optimize this kernel
JitConstants ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw::GetJitConstants(const convolution_params& params,
                                                                      const DispatchData& runInfo) const {
    auto jit = Parent::GetJitConstants(params, runInfo);

    if (!params.fused_ops.empty()) {
        auto input_dt = GetActivationType(params);
        FusedOpsConfiguration conf_scalar = {"", {"b", "f", "y", "x"}, "res", input_dt, 1 };
        jit.Merge(MakeFusedOpsJitConstants(params, {conf_scalar}));
    }

    return jit;
}


KernelsData ConvolutionKernel_MMAD_b_fs_yx_fsv32_dw::GetKernelsData(const Params& params,
                                                                    const optional_params& options) const {
    KernelsData kd = GetTunedKernelsDataByIndex(params, options);
    if (!kd.empty())
        kd[0].estimatedTime = FORCE_PRIORITY_3;
    return kd;
}

}  // namespace kernel_selector
