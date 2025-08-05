// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <map>
#include <shared_mutex>
#include <string>
#include <vector>

#include "openvino/template/compiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::template_plugin::cache {

// Engine-scoped KV Cache Manager: shared by all InferRequests of a TemplateCompiledModel.
class OPENVINO_API CacheManager {
    // Thread-safety for multi-stream use
    mutable std::shared_mutex m_mu;

    size_t m_num_decoder_layers = 0;
    std::string m_device;
    size_t m_block_size = 0;
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks = 0;
    size_t m_block_size_in_bytes = 0;
    ov::IInferRequest m_request;
    ov::IRemoteContext m_context;

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks);
    void update_request_tensor(size_t decoder_layer_id);

public:
    explicit CacheManager(ov::IInferRequest request);

    size_t get_num_decoder_layers() const;
    std::string get_device() const;
    size_t get_block_size() const;
    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const;
    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const;
    size_t get_block_size_in_bytes() const;
    size_t sub_byte_data_type_multiplier(const ov::element::Type data_type) const;

    // Allocates or grows KV storage to hold at least num_kv_blocks pages.
    void allocate_cache_if_needed(size_t num_kv_blocks);

    // Accessors (copy-by-value ov::Tensor handle; safe and cheap)
    ov::Tensor get_key_cache(size_t decoder_layer_id) const;
    ov::Tensor get_value_cache(size_t decoder_layer_id) const;

    // Bulk copy pages (src_block_id -> many dests) across all layers.
    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map);
};

}  // namespace ov::template_plugin::cache
