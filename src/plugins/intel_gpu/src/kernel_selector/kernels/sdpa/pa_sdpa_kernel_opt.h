// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {

struct pa_sdpa_params : base_params {
    pa_sdpa_params() : base_params(KernelType::PA_SDPA) {}

    sdpa_configuration conf;
    size_t max_context_len = 0;
};

class PagedAttentionSDPAKernelOpt : public KernelBaseOpenCL {
public:
    struct DispatchData : CommonDispatchData {
        size_t max_sequence_length;
    };

    PagedAttentionSDPAKernelOpt() : KernelBaseOpenCL{"pa_sdpa_opt"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const pa_sdpa_params& kernel_params) const;
    static CommonDispatchData SetDefault(const pa_sdpa_params& kernel_params, size_t kernel_idx = 0);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
