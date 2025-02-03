// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {

dnnl::reorder getReorderPrim(const MultiCachePtr& cache,
                             const dnnl::engine& engine,
                             const dnnl::memory::desc& src,
                             const dnnl::memory::desc& dest);

}  // namespace intel_cpu
}  // namespace ov