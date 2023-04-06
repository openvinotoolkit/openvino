// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

#include <memory>

namespace ov {
namespace intel_cpu {

dnnl::reorder getReorderPrim(MultiCachePtr cache,
                             const dnnl::engine& engine,
                             const dnnl::memory::desc& src,
                             const dnnl::memory::desc& dest);

}  // namespace intel_cpu
}  // namespace ov