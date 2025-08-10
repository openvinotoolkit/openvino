// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace cache {

/**
 * @brief Engine-scoped KV cache owner/allocator.
 * Thread-safe. Does NOT bind tensors to any InferRequest.
 */
class CacheManager {
public:
    // Construct from runtime model & engine metadata (no request coupling)
    CacheManager(const std::shared_ptr<const ov::Model>& runtime_model,
                 const std::vector<std::string>& execution_devices,
                 const ov::SoPtr<ov::IRemoteContext>& context);

    size_t get_num_decoder_layers() const;
    std::string get_device() const;
    size_t get_block_size() const;

    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const;
    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const;

    size_t get_block_size_in_bytes() const;

    // For packed nibble types
    static size_t sub_byte_data_type_multiplier(const ov::element::Type data_type);

    // Ensure at least num_kv_blocks pages exist (grow-only)
    void allocate_cache_if_needed(size_t num_kv_blocks);

    // Read-only handles to engine-owned KV tensors
    ov::Tensor get_key_cache(size_t decoder_layer_id) const;
    ov::Tensor get_value_cache(size_t decoder_layer_id) const;

    // Utility
    size_t get_v_head_size(size_t layer_id) const;

    // In-place copy single-page chunks K/V: src_block -> dst_blocks (per layer)
    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map);

private:
    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks);

    // immutable after ctor
    size_t m_num_decoder_layers = 0;
    std::string m_device;
    size_t m_block_size = 0;
    ov::SoPtr<ov::IRemoteContext> m_context;

    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;

    // resized by allocate_cache_if_needed()
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0;
    size_t m_block_size_in_bytes = 0;

    size_t m_k_head_size = 0;

    // protect growth / copies
    mutable std::mutex m_mutex;
};

}  // namespace cache
}  // namespace ov
