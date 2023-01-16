// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <dnnl_types.h>
#include "graph_context.h"

#include <common/primitive_hashing_utils.hpp>
namespace ov {
namespace intel_cpu {

dnnl::engine GraphContext::eng(dnnl::engine::kind::cpu, 0);

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
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

dnnl::reorder GraphContext::getReorderPrim(const dnnl::memory::desc& src, const dnnl::memory::desc& dest) const {
    auto builder = [this](const ReorderKey& key) {
        dnnl::primitive_attr attr;
        //DEBUG_LOG(key.src, "->", key.dest);
        dnnl::reorder::primitive_desc pd = dnnl::reorder::primitive_desc(eng, key.src, eng, key.dest, attr, true);
        if (!pd) {
            return dnnl::reorder();
        }
        return dnnl::reorder(pd);
    };

    ReorderKey key = {src, dest};
    if (rtParamsCache) {
        auto result = rtParamsCache->getOrCreate(key, builder);
        return result.first;
    }
    return builder(key);
}

}   // namespace intel_cpu
}   // namespace ov
