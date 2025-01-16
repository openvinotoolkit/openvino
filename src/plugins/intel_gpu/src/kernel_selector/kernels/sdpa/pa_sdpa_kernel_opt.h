// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {

struct pa_sdpa_params : base_params {
    pa_sdpa_params() : base_params(KernelType::PA_SDPA) {}

    bool multi_tokens_mode = false;
    size_t max_context_len = 0;
    sdpa_configuration conf;
};

class PagedAttentionSDPAKernelOpt : public KernelBaseOpenCL {
public:
    PagedAttentionSDPAKernelOpt() : KernelBaseOpenCL{"pa_sdpa_opt"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    virtual ~PagedAttentionSDPAKernelOpt() {}

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const pa_sdpa_params& kernel_params, size_t kernel_idx) const;
    static CommonDispatchData SetDefault(const pa_sdpa_params& kernel_params, size_t kernel_idx = 0);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
