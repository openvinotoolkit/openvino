// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node.h"

namespace ov {
namespace intel_cpu {

std::pair<dnnl::reorder, CacheEntryBase::LookUpStatus> getReorderPrim(MultiCachePtr cache,
                             const dnnl::engine& engine,
                             const dnnl::memory::desc& src,
                             const dnnl::memory::desc& dest);

}  // namespace intel_cpu
}  // namespace ov
