// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cache_manager.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/remote_tensor.hpp"
namespace ov {
namespace cache {

static bool is_gpu_device(const std::string& d) {
    return d.find("GPU") != std::string::npos;
}

CacheManager::CacheManager(const std::shared_ptr<const ov::Model>& runtime_model,
                           const std::vector<std::string>& execution_devices,
                           const ov::SoPtr<ov::IRemoteContext>& context) {
    const bool all_gpu_device = std::all_of(execution_devices.begin(), execution_devices.end(), is_gpu_device);
    OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                    "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
    m_device = execution_devices.empty() ? std::string{} : execution_devices[0];

    // Suggested defaults (match pre-existing logic)
    m_block_size = all_gpu_device ? 16 : 32;

    if (all_gpu_device && context) {
        m_context = context;
    }

    for (const auto& input : runtime_model->inputs()) {
        for (const auto& name : input.get_names()) {
            auto cache_precision = input.get_element_type();
            ov::PartialShape pshape;

            if (name.find("key_cache.") == 0) {
                pshape = input.get_partial_shape();
                m_block_size_in_bytes +=
                    pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                m_key_shapes.push_back(pshape);
                m_key_precisions.push_back(cache_precision);
                break;
            } else if (name.find("value_cache.") == 0) {
                pshape = input.get_partial_shape();
                m_block_size_in_bytes +=
                    pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() * cache_precision.size();
                m_value_shapes.push_back(pshape);
                m_value_precisions.push_back(cache_precision);
                break;
            }
        }
    }

    m_num_decoder_layers = m_value_precisions.size();
    OPENVINO_ASSERT(m_num_decoder_layers == m_key_precisions.size(),
                    "Invalid case: a different number of K and V caches in the model");

    // derive head sizes from shapes (assume [B, H, page, D_k/v])
    if (!m_key_shapes.empty()) {
        m_k_head_size = m_key_shapes[0][3].get_length();
    }
}

ov::Shape CacheManager::set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
    pshape[0] = num_kv_blocks;
    return pshape.get_shape();
}

size_t CacheManager::get_num_decoder_layers() const {
    return m_num_decoder_layers;
}
std::string CacheManager::get_device() const {
    return m_device;
}
size_t CacheManager::get_block_size() const {
    return m_block_size;
}

ov::element::Type CacheManager::get_key_cache_precision(size_t decoder_layer_id) const {
    OPENVINO_ASSERT(decoder_layer_id < m_key_precisions.size());
    return m_key_precisions[decoder_layer_id];
}
ov::element::Type CacheManager::get_value_cache_precision(size_t decoder_layer_id) const {
    OPENVINO_ASSERT(decoder_layer_id < m_value_precisions.size());
    return m_value_precisions[decoder_layer_id];
}

size_t CacheManager::get_block_size_in_bytes() const {
    return m_block_size_in_bytes;
}

size_t CacheManager::sub_byte_data_type_multiplier(const ov::element::Type data_type) {
    if (data_type == ov::element::i4 || data_type == ov::element::u4)
        return 2;
    return 1;
}

void CacheManager::allocate_cache_if_needed(size_t num_kv_blocks) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_num_allocated_kv_blocks >= num_kv_blocks)
        return;

    m_num_allocated_kv_blocks = num_kv_blocks;

    for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
        ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);
        ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);

        ov::Tensor key_cache, value_cache;
        if (m_context) {
            auto key_remote   = m_context->create_tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
            auto value_remote = m_context->create_tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);
            key_cache   = ov::make_tensor(key_remote);
            value_cache = ov::make_tensor(value_remote);
        } else {
            key_cache = ov::Tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
            value_cache = ov::Tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);
        }

        if (decoder_layer_id < m_key_cache.size()) {
            auto copy_roi = [&](ov::Tensor& dst, const ov::Tensor& src) {
                if (dst.get_element_type() == ov::element::u4 || dst.get_element_type() == ov::element::i4) {
                    size_t size = src.get_size();
                    std::memcpy(dst.data(), src.data(), size / sub_byte_data_type_multiplier(dst.get_element_type()));
                } else {
                    ov::Tensor roi(dst, ov::Coordinate{0, 0, 0, 0}, src.get_shape());
                    src.copy_to(roi);
                }
            };
            copy_roi(key_cache, m_key_cache[decoder_layer_id]);
            copy_roi(value_cache, m_value_cache[decoder_layer_id]);

            m_key_cache[decoder_layer_id] = key_cache;
            m_value_cache[decoder_layer_id] = value_cache;
        } else {
            m_key_cache.push_back(key_cache);
            m_value_cache.push_back(value_cache);
        }
    }
}

ov::Tensor CacheManager::get_key_cache(size_t decoder_layer_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
    return m_key_cache[decoder_layer_id];
}
ov::Tensor CacheManager::get_value_cache(size_t decoder_layer_id) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
    return m_value_cache[decoder_layer_id];
}

size_t CacheManager::get_v_head_size(size_t) const {
    return !m_value_shapes.empty() ? m_value_shapes[0][3].get_length() : m_k_head_size;
}

void CacheManager::copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& [src_block_id, dst_block_ids] : block_copy_map) {
        for (size_t dst_block_id : dst_block_ids) {
            for (size_t layer = 0; layer < m_num_decoder_layers; ++layer) {
                auto copy_one_block =
                    [&](ov::Tensor& dst, const ov::Tensor& src, size_t src_start, size_t dst_start, size_t stride) {
                        const bool is_remote = dst.is<ov::RemoteTensor>() || src.is<ov::RemoteTensor>();
                        if (!is_remote) {
                            const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(src.data()) + src_start * stride;
                            uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.data()) + dst_start * stride;
                            std::memcpy(dst_ptr, src_ptr, stride);
                        }
                    };

                const auto& key_shape = m_key_cache[layer].get_shape();
                const auto& val_shape = m_value_cache[layer].get_shape();

                size_t stride_k =
                    std::accumulate(key_shape.begin() + 1, key_shape.end(), 1, std::multiplies<size_t>()) /
                    sub_byte_data_type_multiplier(m_key_cache[layer].get_element_type());

                size_t stride_v =
                    std::accumulate(val_shape.begin() + 1, val_shape.end(), 1, std::multiplies<size_t>()) /
                    sub_byte_data_type_multiplier(m_value_cache[layer].get_element_type());

                copy_one_block(m_key_cache[layer], m_key_cache[layer], src_block_id, dst_block_id, stride_k);
                copy_one_block(m_value_cache[layer], m_value_cache[layer], src_block_id, dst_block_id, stride_v);
            }
        }
    }
}

}  // namespace cache
}  // namespace ov
