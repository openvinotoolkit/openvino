// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_prim.h"

#include <common/utils.hpp>
#include <cstddef>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_common.hpp>

#include "cache/multi_cache.h"
#include "common/primitive_hashing_utils.hpp"
#include "utils/debug_capabilities.h"

namespace ov::intel_cpu {

struct ReorderKey {
    dnnl::memory::desc src;
    dnnl::memory::desc dest;
    [[nodiscard]] size_t hash() const;
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

dnnl::reorder getReorderPrim(const MultiCachePtr& cache,
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
        auto result = cache->getOrCreate(key, builder);
        return result.first;
    }
    return builder(key);
}

}  // namespace ov::intel_cpu
