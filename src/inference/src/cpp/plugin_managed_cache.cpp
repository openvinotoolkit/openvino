// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/plugin_managed_cache.hpp"

#include "openvino/core/except.hpp"
#include "openvino/runtime/iplugin_managed_cache.hpp"

#define OV_PLUGIN_MANAGED_CACHE_CALL(...)                                                     \
    OPENVINO_ASSERT(_impl != nullptr, "PluginManagedCache was not initialized.");             \
    try {                                                                                     \
        __VA_ARGS__;                                                                          \
    } catch (const std::exception& ex) {                                                      \
        OPENVINO_THROW(ex.what());                                                            \
    } catch (...) {                                                                           \
        OPENVINO_THROW("Unexpected exception");                                               \
    }

namespace ov {

IPluginManagedCache::~IPluginManagedCache() = default;

PluginManagedCache::PluginManagedCache(std::shared_ptr<IPluginManagedCache> impl) : _impl{std::move(impl)} {
    OPENVINO_ASSERT(_impl != nullptr, "PluginManagedCache was not initialized.");
}

PluginManagedCache::operator bool() const noexcept {
    return _impl != nullptr;
}

std::vector<CacheGroupInfo> PluginManagedCache::get_groups() const {
    OV_PLUGIN_MANAGED_CACHE_CALL(return _impl->get_groups());
}

size_t PluginManagedCache::get_available_memory() const {
    OV_PLUGIN_MANAGED_CACHE_CALL(return _impl->get_available_memory());
}

void PluginManagedCache::allocate(size_t group, size_t num_blocks, bool zero_initialize) {
    OV_PLUGIN_MANAGED_CACHE_CALL(_impl->allocate(group, num_blocks, zero_initialize));
}

void PluginManagedCache::reallocate(size_t group, size_t num_blocks, bool zero_initialize_new_blocks) {
    OV_PLUGIN_MANAGED_CACHE_CALL(_impl->reallocate(group, num_blocks, zero_initialize_new_blocks));
}

void PluginManagedCache::deallocate(size_t group) {
    OV_PLUGIN_MANAGED_CACHE_CALL(_impl->deallocate(group));
}

size_t PluginManagedCache::get_num_allocated_blocks(size_t group) const {
    OV_PLUGIN_MANAGED_CACHE_CALL(return _impl->get_num_allocated_blocks(group));
}

void PluginManagedCache::copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& block_copy_map) {
    OV_PLUGIN_MANAGED_CACHE_CALL(_impl->copy_blocks(group, block_copy_map));
}

void PluginManagedCache::zero_blocks(size_t group, const std::set<size_t>& block_indices) {
    OV_PLUGIN_MANAGED_CACHE_CALL(_impl->zero_blocks(group, block_indices));
}

}  // namespace ov
