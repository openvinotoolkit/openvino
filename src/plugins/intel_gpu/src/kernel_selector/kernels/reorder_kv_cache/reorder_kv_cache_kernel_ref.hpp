// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {

struct reorder_kv_cache_params : base_params {
    reorder_kv_cache_params() : base_params(KernelType::REORDER_KV_CACHE) {}
    uint32_t seq_len = 0;
    int64_t indirect_axis = 0;
};

class ReorderKVCacheKernelRef : public KernelBaseOpenCL {
public:
    ReorderKVCacheKernelRef() : KernelBaseOpenCL{"reorder_kv_cache_ref"} {}
    KernelsData GetKernelsData(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;

protected:
    bool Validate(const Params& params) const override;
    JitConstants GetJitConstants(const reorder_kv_cache_params& kernel_params) const;
    static CommonDispatchData SetDefault(const reorder_kv_cache_params& kernel_params);
    void GetUpdateDispatchDataFunc(KernelData& kd) const override;
};

}  // namespace kernel_selector
