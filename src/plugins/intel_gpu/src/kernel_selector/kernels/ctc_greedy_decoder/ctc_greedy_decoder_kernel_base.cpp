// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
            MakeJitConstant("N_", inp.Batch().v),
            MakeJitConstant("T_", inp.Feature().v)
        });
    } else {
        jit.AddConstants({
            MakeJitConstant("T_", inp.Batch().v),
            MakeJitConstant("N_", inp.Feature().v)
        });
    }

    return jit;
}

CTCGreedyDecoderKernelBase::DispatchData CTCGreedyDecoderKernelBase::SetDefault(const ctc_greedy_decoder_params& params) const {
    DispatchData dispatchData;

    dispatchData.gws = { 1, 1, 1 };
    dispatchData.lws = GetOptimalLocalWorkGroupSizes(dispatchData.gws, params.engineInfo);

    return dispatchData;
}

KernelsData CTCGreedyDecoderKernelBase::GetCommonKernelsData(const Params& params) const {
    assert(params.GetType() == KernelType::CTC_GREEDY_DECODER);

    if (!Validate(params))
        return {};

    const ctc_greedy_decoder_params& orgParams = static_cast<const ctc_greedy_decoder_params&>(params);

    DispatchData dispatchData = SetDefault(orgParams);

    KernelData kd = KernelData::Default<ctc_greedy_decoder_params>(params);

    auto cldnn_jit = GetJitConstants(orgParams, dispatchData);
    auto entry_point = GetEntryPoint(kernelName, orgParams.layerID, params);
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
        kernel.params.arguments.push_back({ArgumentDescriptor::Types::OUTPUT, 1});
    }

    return {kd};
}

}  // namespace kernel_selector
