// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Helpers to attach/retrieve a PagedCacheManager handle on PA nodes via rt_info.

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/reference/utils/paged_cache_manager.hpp"

namespace ov {
namespace reference {
namespace paged_attention_cache {

/// rt_info key for the cache manager handle.
inline constexpr const char* CACHE_MANAGER_RT_INFO_KEY = "__paged_cache_manager";

using CacheManagerHandle = std::shared_ptr<void>;

/// Create a type-erased PagedCacheManager handle.
inline CacheManagerHandle make_cache_handle(ov::element::Type et) {
    auto* mgr = new PagedCacheManager(et);
    return CacheManagerHandle(static_cast<void*>(mgr), [](void* p) {
        delete static_cast<PagedCacheManager*>(p);
    });
}

/// Store a cache manager handle on a node.
inline void set_cache_manager(ov::Node* node, CacheManagerHandle handle) {
    node->get_rt_info()[CACHE_MANAGER_RT_INFO_KEY] = std::move(handle);
}

/// Get the cache manager handle from a node, or nullptr if none.
inline CacheManagerHandle get_cache_manager(const ov::Node* node) {
    auto& rt = node->get_rt_info();
    auto it = rt.find(CACHE_MANAGER_RT_INFO_KEY);
    if (it == rt.end()) {
        return nullptr;
    }
    return it->second.as<CacheManagerHandle>();
}

/// Get the concrete PagedCacheManager pointer, or nullptr.
inline PagedCacheManager* get_cache_manager_ptr(const ov::Node* node) {
    auto h = get_cache_manager(node);
    return h ? static_cast<PagedCacheManager*>(h.get()) : nullptr;
}

}  // namespace paged_attention_cache
}  // namespace reference
}  // namespace ov
