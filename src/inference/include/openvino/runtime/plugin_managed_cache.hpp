// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Caches whose storage a plugin manages, and the compiled-model property that hands them out.
 * @file openvino/runtime/plugin_managed_cache.hpp
 */

#pragma once

#include <cstddef>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief Kind of a plugin-managed cache group.
 * @ingroup ov_runtime_cpp_api
 */
enum class CacheKind {
    KEY_VALUE,               ///< paged-attention KV cache: key_cache.N / value_cache.N
    LINEAR_ATTENTION_STATE,  ///< fixed-size per-sequence state: <prefix>_state_table.N
};

/**
 * @brief One model input whose storage the plugin manages, in the model-level representation.
 * @ingroup ov_runtime_cpp_api
 */
struct CacheTensorInfo {
    std::string name;               ///< e.g. "key_cache.3"
    size_t layer = 0;
    ov::element::Type element_type;
    ov::PartialShape shape;         ///< as the compiled model reports the input; dimension 0: blocks (dynamic)
};

/**
 * @brief One cache kind with its own block pool.
 * @ingroup ov_runtime_cpp_api
 */
struct CacheGroupInfo {
    CacheKind kind = CacheKind::KEY_VALUE;
    std::vector<CacheTensorInfo> tensors;  ///< all tensors of the group
    size_t block_size = 0;                 ///< tokens per block (KEY_VALUE); 1 for LINEAR_ATTENTION_STATE
    size_t block_size_in_bytes = 0;        ///< one block of all tensors of the group, model-level
};

class IPluginManagedCache;

/**
 * @brief Caches whose storage the plugin manages, e.g. because they are split across devices.
 *
 * Returned by the compiled-model property ov::plugin_managed_cache, present only if the plugin manages the
 * caches; otherwise the application allocates and binds the cache inputs itself. The facade must not outlive
 * the compiled model it was obtained from.
 *
 * Everything is model-level: layouts, sizes and block indices refer to the model as if it were not split. A
 * block index addresses the same tokens (KEY_VALUE) or the same sequence slot (LINEAR_ATTENTION_STATE) in every
 * layer of the group, and an operation applies to all tensors of the group. An operation's effect is visible to
 * the next inference when it returns; operations must not overlap an inference.
 * @ingroup ov_runtime_cpp_api
 */
class OPENVINO_RUNTIME_API PluginManagedCache {
public:
    PluginManagedCache() = default;

    /// @brief Wraps a plugin's implementation (dev API type).
    explicit PluginManagedCache(std::shared_ptr<IPluginManagedCache> impl);

    /// @brief Whether the facade holds an implementation.
    explicit operator bool() const noexcept;

    /// @brief The groups the plugin manages, model-level.
    std::vector<CacheGroupInfo> get_groups() const;

    /// @brief Memory, in bytes, available to all groups together.
    size_t get_available_memory() const;

    /// @brief Allocates `num_blocks` blocks for a group that holds none.
    void allocate(size_t group, size_t num_blocks, bool zero_initialize);

    /// @brief Grows or shrinks a group's storage; blocks [0, min(old, new)) keep their contents.
    ///
    /// Failure leaves the previous storage, its size and its contents unchanged.
    void reallocate(size_t group, size_t num_blocks, bool zero_initialize_new_blocks);

    /// @brief Frees a group's storage.
    void deallocate(size_t group);

    /// @brief Blocks the group holds now; 0 before allocate() and after deallocate().
    size_t get_num_allocated_blocks(size_t group) const;

    /// @brief Copies blocks within a group: source block -> destination blocks.
    void copy_blocks(size_t group, const std::map<size_t, std::list<size_t>>& block_copy_map);

    /// @brief Zeroes the listed blocks of a group.
    void zero_blocks(size_t group, const std::set<size_t>& block_indices);

private:
    std::shared_ptr<IPluginManagedCache> _impl;
};

/**
 * @brief Read-only compiled-model property, present only when the plugin manages the caches.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<PluginManagedCache, PropertyMutability::RO> plugin_managed_cache{"PLUGIN_MANAGED_CACHE"};

}  // namespace ov
