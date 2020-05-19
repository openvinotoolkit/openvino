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


#include "ctc_greedy_decoder_kernel_base.h"
#include "kernel_selector_utils.h"
#include <vector>

namespace kernel_selector {
JitConstants CTCGreedyDecoderKernelBase::GetJitConstants(const ctc_greedy_decoder_params& params, CTCGreedyDecoderKernelBase::DispatchData) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);
    auto inp = params.inputs[0];

    jit.AddConstants({
        MakeJitConstant("ctc_merge_repeated_", params.merge_repeated),
        MakeJitConstant("T_", inp.Batch().v),
        MakeJitConstant("N_", inp.Feature().v),
        MakeJitConstant("C_", inp.Y().v)
    });

    return jit;
}

CTCGreedyDecoderKernelBase::DispatchData CTCGreedyDecoderKernelBase::SetDefault(const ctc_greedy_decoder_params& params) const {
    DispatchData kd;
    kd.fp16UnitUsed = params.inputs[0].GetDType() == Datatype::F16;

    std::vector<size_t> global = { 1, 1, 1 };
    auto local = GetOptimalLocalWorkGroupSizes(global, params.engineInfo);

    kd.gws0 = global[0];
    kd.gws1 = global[1];
    kd.gws2 = global[2];

    kd.lws0 = local[0];
    kd.lws1 = local[1];
    kd.lws2 = local[2];

    return kd;
}

KernelsData CTCGreedyDecoderKernelBase::GetCommonKernelsData(const Params& params,
                                                const optional_params& options,
                                                float estimated_time) const {
    assert(params.GetType() == KernelType::CTC_GREEDY_DECODER);

    if (!Validate(params, options))
        return {};

    const ctc_greedy_decoder_params& orgParams = static_cast<const ctc_greedy_decoder_params&>(params);

    DispatchData runInfo;

    runInfo = SetDefault(orgParams);

    KernelData kd = KernelData::Default<ctc_greedy_decoder_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, runInfo);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     runInfo,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2,  // input and sequence indicatiors
                     GetFusedPrimitiveInputsCount(params));

    kd.estimatedTime = estimated_time;

    return {kd};
}

}  // namespace kernel_selector
