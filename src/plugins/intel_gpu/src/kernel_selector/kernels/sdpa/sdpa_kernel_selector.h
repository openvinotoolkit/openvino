// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class sdpa_kernel_selector : public kernel_selector_base {
public:
    static sdpa_kernel_selector& Instance() {
        static sdpa_kernel_selector instance_;
        return instance_;
    }

    sdpa_kernel_selector();

    virtual ~sdpa_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};

class kv_cache_update_kernel_selector : public kernel_selector_base {
public:
    static kv_cache_update_kernel_selector& Instance() {
        static kv_cache_update_kernel_selector instance_;
        return instance_;
    }

    kv_cache_update_kernel_selector();

    virtual ~kv_cache_update_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};

class kv_cache_rotate_kernel_selector : public kernel_selector_base {
public:
    static kv_cache_rotate_kernel_selector& Instance() {
        static kv_cache_rotate_kernel_selector instance_;
        return instance_;
    }

    kv_cache_rotate_kernel_selector();

    virtual ~kv_cache_rotate_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};

class pa_sdpa_kernel_selector : public kernel_selector_base {
public:
    static pa_sdpa_kernel_selector& Instance() {
        static pa_sdpa_kernel_selector instance_;
        return instance_;
    }

    pa_sdpa_kernel_selector();

    virtual ~pa_sdpa_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector
