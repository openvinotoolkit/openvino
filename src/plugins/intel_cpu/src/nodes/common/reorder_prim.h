// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_common.h>
#include <node.h>

#include <memory>

namespace ov {
namespace intel_cpu {

std::shared_ptr<dnnl::primitive> getReorderPrim(MultiCachePtr cache,
                                                const dnnl::engine& engine,
                                                const dnnl::memory::desc& src,
                                                const dnnl::memory::desc& dest,
                                                impl_desc_type* p_impl_type = nullptr);

inline std::shared_ptr<dnnl::primitive> getReorderPrim(MultiCachePtr cache,
                                                const dnnl::memory& src,
                                                const dnnl::memory& dest,
                                                impl_desc_type* p_impl_type = nullptr) {
    return getReorderPrim(cache, dest.get_engine(), src.get_desc(), dest.get_desc(), p_impl_type);
}

}  // namespace intel_cpu
}  // namespace ov