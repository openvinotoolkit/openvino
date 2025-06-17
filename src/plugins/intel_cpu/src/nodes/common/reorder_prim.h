// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cache/multi_cache.h"

namespace ov::intel_cpu {

dnnl::reorder getReorderPrim(const MultiCachePtr& cache,
                             const dnnl::engine& engine,
                             const dnnl::memory::desc& src,
                             const dnnl::memory::desc& dest);

}  // namespace ov::intel_cpu
