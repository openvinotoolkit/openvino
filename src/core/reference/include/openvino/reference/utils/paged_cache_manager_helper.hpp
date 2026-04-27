// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Helper utilities to attach / retrieve a PagedCacheManager handle
// to / from a PagedAttentionExtension node via its rt_info map.
//
// This keeps the common PagedAttentionExtension op free of any
// cache-management fields while allowing the template plugin (and tests)
// to store a shared handle on each node.

#pragma once

#include <memory>

#include "openvino/core/node.hpp"
#include "openvino/reference/utils/paged_cache_manager.hpp"

namespace ov {
namespace reference {
namespace paged_attention_cache {

/// rt_info key under which the cache manager handle is stored.
inline constexpr const char* CACHE_MANAGER_RT_INFO_KEY = "__paged_cache_manager";

using CacheManagerHandle = std::shared_ptr<void>;

/// Create a new PagedCacheManager wrapped in a type-erased handle.
inline CacheManagerHandle make_cache_handle(ov::element::Type et) {
    auto* mgr = new PagedCacheManager(et);
    return CacheManagerHandle(static_cast<void*>(mgr), [](void* p) {
        delete static_cast<PagedCacheManager*>(p);
    });
}

/// Attach a cache manager handle to a node.
inline void set_cache_manager(ov::Node* node, CacheManagerHandle handle) {
    node->get_rt_info()[CACHE_MANAGER_RT_INFO_KEY] = std::move(handle);
}

/// Retrieve the cache manager handle from a node (returns nullptr if none).
inline CacheManagerHandle get_cache_manager(const ov::Node* node) {
    auto& rt = node->get_rt_info();
    auto it = rt.find(CACHE_MANAGER_RT_INFO_KEY);
    if (it == rt.end()) {
        return nullptr;
    }
    return it->second.as<CacheManagerHandle>();
}

/// Convenience: retrieve and cast to the concrete PagedCacheManager*.
inline PagedCacheManager* get_cache_manager_ptr(const ov::Node* node) {
    auto h = get_cache_manager(node);
    return h ? static_cast<PagedCacheManager*>(h.get()) : nullptr;
}

}  // namespace paged_attention_cache
}  // namespace reference
}  // namespace ov
