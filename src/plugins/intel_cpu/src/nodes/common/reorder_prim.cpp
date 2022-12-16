// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_prim.h"

#include <dnnl_extension_utils.h>
#include <dnnl_types.h>

#include <algorithm>
#include <common/primitive_hashing_utils.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
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
    seed = hash_combine(seed, get_md_hash(src.data));
    seed = hash_combine(seed, get_md_hash(dest.data));

    return seed;
}

bool ReorderKey::operator==(const ReorderKey& rhs) const {
    bool retVal = true;
    retVal = src == rhs.src && dest == rhs.dest;
    return retVal;
}

std::shared_ptr<dnnl::primitive> getReorderPrim(MultiCachePtr cache,
                                                const dnnl::engine& engine,
                                                const dnnl::memory::desc& src,
                                                const dnnl::memory::desc& dest,
                                                impl_desc_type* p_impl_type) {
    auto builder = [&engine, &p_impl_type](const ReorderKey& key) -> std::shared_ptr<dnnl::primitive> {
        dnnl::primitive_attr attr;
        DEBUG_LOG(key.src, "->", key.dest);
        dnnl::reorder::primitive_desc pd = dnnl::reorder::primitive_desc(engine, key.src, engine, key.dest, attr, true);
        if (!pd)
            return nullptr;
        auto info = pd.impl_info_str();
        if (p_impl_type)
            *p_impl_type = parse_impl_name(info);
        return std::make_shared<dnnl::reorder>(pd);
    };

    ReorderKey key = {src, dest};
    if (cache) {
        auto result = cache->getOrCreate(key, builder);
        return result.first;
    }
    return builder(key);
}

}  // namespace intel_cpu
}  // namespace ov