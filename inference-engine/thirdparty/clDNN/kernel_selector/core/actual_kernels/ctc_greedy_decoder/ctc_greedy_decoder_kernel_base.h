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
// ctc_greedy_decoder_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct ctc_greedy_decoder_optional_params : optional_params {
    ctc_greedy_decoder_optional_params() : optional_params(KernelType::CTC_GREEDY_DECODER) {}
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
    KernelsData GetCommonKernelsData(const Params& params, const optional_params&) const;
};
}  // namespace kernel_selector
