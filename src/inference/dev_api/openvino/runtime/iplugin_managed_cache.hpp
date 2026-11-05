// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Interface a plugin implements to manage cache storage itself.
 * @file openvino/runtime/iplugin_managed_cache.hpp
 */

#pragma once

#include "openvino/runtime/plugin_managed_cache.hpp"

namespace ov {

/**
 * @brief Plugin side of ov::PluginManagedCache; called only by the facade.
 *
 * A plugin returns its implementation wrapped in the facade as the value of ov::plugin_managed_cache.
 * The semantics of every method are those of the facade method of the same name; errors are thrown.
 */
class OPENVINO_RUNTIME_API IPluginManagedCache {
public:
    virtual ~IPluginManagedCache();

    virtual std::vector<CacheGroupInfo> get_groups() const = 0;
    virtual size_t get_available_memory() const = 0;
    virtual void allocate(size_t group, size_t num_blocks, bool zero_initialize) = 0;
    virtual void reallocate(size_t group, size_t num_blocks, bool zero_initialize_new_blocks) = 0;
    virtual void deallocate(size_t group) = 0;
    virtual size_t get_num_allocated_blocks(size_t group) const = 0;
    virtual void copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& block_copy_map) = 0;
    virtual void zero_blocks(size_t group, const std::set<size_t>& block_indices) = 0;
};

}  // namespace ov
