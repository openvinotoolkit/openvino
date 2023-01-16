// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_prim.h"

#include "dnnl_extension_utils.h"
#include "dnnl_types.h"

#include <algorithm>
#include "common/primitive_hashing_utils.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include <memory>
#include <string>

#include "utils/general_utils.h"

namespace ov {
namespace intel_cpu {

struct ReorderKey {
    dnnl::memory::desc src;
    dnnl::memory::desc dest;
    size_t hash() const;
    bool operator==(const ReorderKey& rhs) const;
};

size_t ReorderKey::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    seed = hash_combine(seed, get_md_hash(*src.get()));
    seed = hash_combine(seed, get_md_hash(*dest.get()));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

std::pair<dnnl::reorder, CacheEntryBase::LookUpStatus> getReorderPrim(MultiCachePtr cache,
                                                                      const dnnl::engine& engine,
                                                                      const dnnl::memory::desc& src,
                                                                      const dnnl::memory::desc& dest) {
    auto builder = [&engine](const ReorderKey& key) {
        dnnl::primitive_attr attr;
        DEBUG_LOG(key.src, "->", key.dest);
        dnnl::reorder::primitive_desc pd = dnnl::reorder::primitive_desc(engine, key.src, engine, key.dest, attr, true);
        if (!pd) {
            return dnnl::reorder();
        }
        return dnnl::reorder(pd);
    };

    ReorderKey key = {src, dest};
    if (cache) {
        return cache->getOrCreate(key, builder);
    }

    return {builder(key), CacheEntryBase::LookUpStatus::Miss};
}

}  // namespace intel_cpu
}  // namespace ov
