// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kv_cache_kernel_selector.hpp"

#include "reorder_kv_cache_kernel_ref.hpp"

namespace kernel_selector {

reorder_kv_cache_kernel_selector::reorder_kv_cache_kernel_selector() {
    Attach<ReorderKVCacheKernelRef>();
}

KernelsData reorder_kv_cache_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::REORDER_KV_CACHE);
}

reorder_kv_cache_kernel_selector& reorder_kv_cache_kernel_selector::Instance() {
    static reorder_kv_cache_kernel_selector instance;
    return instance;
}

}  // namespace kernel_selector
