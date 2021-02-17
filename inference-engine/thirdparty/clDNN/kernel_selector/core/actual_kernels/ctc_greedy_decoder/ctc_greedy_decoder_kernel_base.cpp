// Copyright (c) 2020-2021 Intel Corporation
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
        MakeJitConstant("blank_index_", params.blank_index),
        MakeJitConstant("C_", inp.Y().v)
    });

    if (params.outputs_num == 2) {
        jit.AddConstants({
            MakeJitConstant("SECOND_OUTPUT_EXIST", 1),
            MakeJitConstant("N_", inp.Batch().v),
            MakeJitConstant("T_", inp.Feature().v)
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("T_", inp.Batch().v),
            MakeJitConstant("N_", inp.Feature().v)
        });
    };

    return jit;
}

CTCGreedyDecoderKernelBase::DispatchData CTCGreedyDecoderKernelBase::SetDefault(const ctc_greedy_decoder_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws = { 1, 1, 1 };
    dispatchData.lws= GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData CTCGreedyDecoderKernelBase::GetCommonKernelsData(const Params& params,
                                                             const optional_params& options) const {
    assert(params.GetType() == KernelType::CTC_GREEDY_DECODER);

    if (!Validate(params, options))
        return {};

    const ctc_greedy_decoder_params& orgParams = static_cast<const ctc_greedy_decoder_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<ctc_greedy_decoder_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, options);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];
    FillCLKernelData(kernel,
                     dispatchData,
                     params.engineInfo,
                     kernelName,
                     jit,
                     entry_point,
                     "",
                     false,
                     false,
                     2,  // input and sequence indicatiors
                     GetFusedPrimitiveInputsCount(params));

    if (orgParams.outputs_num == 2) {
        kernel.arguments.push_back({ArgumentDescriptor::Types::INPUT, 2});
    }

    return {kd};
}

}  // namespace kernel_selector
