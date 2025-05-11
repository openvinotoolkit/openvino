// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <string>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ctc_greedy_decoder_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ctc_greedy_decoder_params : public base_params {
    ctc_greedy_decoder_params() : base_params(KernelType::CTC_GREEDY_DECODER) {}

    bool merge_repeated = true;
    uint32_t blank_index = 0;
    uint32_t outputs_num = 1;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CTCGreedyDecoderKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class CTCGreedyDecoderKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~CTCGreedyDecoderKernelBase() {}
    using DispatchData = CommonDispatchData;

protected:
    virtual JitConstants GetJitConstants(const ctc_greedy_decoder_params& params, DispatchData dispatchData) const;
    virtual DispatchData SetDefault(const ctc_greedy_decoder_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;
};
}  // namespace kernel_selector
