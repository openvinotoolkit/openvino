// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {

struct kv_cache_update_params : base_params {
    kv_cache_update_params() : base_params(KernelType::PA_KV_CACHE_UPDATE) {}

    bool is_prefill = false;
    sdpa_configuration conf;
};

class KVCacheUpdateKernelRef : public KernelBaseOpenCL {
public:
    KVCacheUpdateKernelRef() : KernelBaseOpenCL{"pa_kv_cache_update_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    virtual ~KVCacheUpdateKernelRef() {}

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const kv_cache_update_params& kernel_params) const;
    static CommonDispatchData SetDefault(const kv_cache_update_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
