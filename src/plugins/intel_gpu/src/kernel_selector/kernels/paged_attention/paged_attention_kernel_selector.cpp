// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_attention_kernel_selector.hpp"
#include "kv_cache_update_kernel_ref.hpp"
#include "sdpa_kernel_ref.hpp"

namespace kernel_selector {

kv_cache_update_kernel_selector::kv_cache_update_kernel_selector() {
    Attach<KVCacheUpdateKernelRef>();
}

KernelsData kv_cache_update_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PA_KV_CACHE_UPDATE);
}

kv_cache_update_kernel_selector& kv_cache_update_kernel_selector::Instance() {
    static kv_cache_update_kernel_selector instance;
    return instance;
}

sdpa_kernel_selector::sdpa_kernel_selector() {
    Attach<SDPAKernelRef>();
}

KernelsData sdpa_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PA_SDPA);
}

sdpa_kernel_selector& sdpa_kernel_selector::Instance() {
    static sdpa_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
