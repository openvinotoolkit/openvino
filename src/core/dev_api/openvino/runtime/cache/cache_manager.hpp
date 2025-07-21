// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <list>
#include <map>
#include <string>
#include <vector>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::cache {

class OPENVINO_API CacheManager {
    size_t m_num_decoder_layers;
    std::string m_device;
    size_t m_block_size;
    std::vector<ov::element::Type> m_key_precisions, m_value_precisions;
    std::vector<ov::PartialShape> m_key_shapes, m_value_shapes;
    std::vector<ov::Tensor> m_key_cache, m_value_cache;
    size_t m_num_allocated_kv_blocks, m_block_size_in_bytes;
    ov::InferRequest m_request;
    size_t m_k_head_size;
    ov::RemoteContext m_context;

    static ov::Shape set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks);
    void update_request_tensor(size_t decoder_layer_id);

public:
    explicit CacheManager(ov::InferRequest request);

    size_t get_num_decoder_layers() const;
    std::string get_device() const;
    size_t get_block_size() const;
    ov::element::Type get_key_cache_precision(size_t decoder_layer_id) const;
    ov::element::Type get_value_cache_precision(size_t decoder_layer_id) const;
    size_t get_block_size_in_bytes() const;
    size_t sub_byte_data_type_multiplier(const ov::element::Type data_type) const;
    void allocate_cache_if_needed(size_t num_kv_blocks);
    ov::Tensor get_key_cache(size_t decoder_layer_id) const;
    ov::Tensor get_value_cache(size_t decoder_layer_id) const;
    size_t get_v_head_size(size_t layer_id) const;
    void copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map);
};

}  // namespace ov::cache
