// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa_kernel_selector.h"
#include "sdpa_kernel_ref.h"
#include "sdpa_kernel_opt.h"
#include "sdpa_kernel_micro.h"

#include "pa_sdpa_kernel_opt.h"
#include "pa_kv_cache_update_kernel_ref.h"
#include "pa_kv_cache_rotate_kernel_ref.h"

namespace kernel_selector {

sdpa_kernel_selector::sdpa_kernel_selector() {
    Attach<SDPAKernelOpt>();
    Attach<SDPAKernelRef>();
#ifdef ENABLE_ONEDNN_FOR_GPU
    Attach<SDPAKernelMicro>();
#endif
}

KernelsData sdpa_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::SDPA);
}

kv_cache_update_kernel_selector::kv_cache_update_kernel_selector() {
    Attach<KVCacheUpdateKernelRef>();
}

KernelsData kv_cache_update_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PA_KV_CACHE_UPDATE);
}

kv_cache_rotate_kernel_selector::kv_cache_rotate_kernel_selector() {
    Attach<KVCacheRotateKernelRef>();
}

KernelsData kv_cache_rotate_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PA_KV_CACHE_ROTATE);
}

pa_sdpa_kernel_selector::pa_sdpa_kernel_selector() {
    Attach<PagedAttentionSDPAKernelOpt>();
}

KernelsData pa_sdpa_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::PA_SDPA);
}
}  // namespace kernel_selector
