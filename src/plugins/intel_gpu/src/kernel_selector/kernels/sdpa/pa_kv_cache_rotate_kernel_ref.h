// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "sdpa_kernel_base.h"

namespace kernel_selector {

struct kv_cache_rotate_params : base_params {
    kv_cache_rotate_params() : base_params(KernelType::PA_KV_CACHE_ROTATE) {}

    Datatype original_cache_dt;
    sdpa_configuration conf;
};

class KVCacheRotateKernelRef : public KernelBaseOpenCL {
public:
    KVCacheRotateKernelRef() : KernelBaseOpenCL{"pa_kv_cache_rotate_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
    virtual ~KVCacheRotateKernelRef() {}

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const kv_cache_rotate_params& kernel_params) const;
    static CommonDispatchData SetDefault(const kv_cache_rotate_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
