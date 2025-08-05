// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/template/cache/cache_manager.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace ov::template_plugin::cache {

CacheManager::CacheManager(ov::IInferRequest request) : m_request(request) {
    ov::CompiledModel compiled_model = request.get_compiled_model();
    std::vector<std::string> execution_devices = compiled_model.get_property(ov::execution_devices);
    const bool all_gpu_device =
        std::all_of(execution_devices.begin(), execution_devices.end(), [](const std::string& device) {
            return device.find("GPU") != std::string::npos;
        });
    OPENVINO_ASSERT(all_gpu_device || execution_devices.size() == 1,
                    "Continuous batching: execution device is expected to be single CPU / single GPU / multi GPUs");
    m_device = execution_devices[0];

    m_block_size = all_gpu_device ? 16 : 32;

    if (all_gpu_device) {
        m_context = compiled_model.get_context();
    }

    for (const auto& input : compiled_model.inputs()) {
        for (const auto& name : input.get_names()) {
            auto cache_precision = input.get_element_type();
            ov::PartialShape pshape;

            if (name.find("key_cache.") == 0) {
                pshape = input.get_partial_shape();
                OPENVINO_ASSERT(pshape.rank().is_static() &&
                                    pshape[1].is_static() && pshape[2].is_static() && pshape[3].is_static(),
                                "KV key cache shape must be static");
                m_block_size_in_bytes +=
                    pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() *
                    cache_precision.size() / sub_byte_data_type_multiplier(cache_precision);
                m_key_shapes.push_back(pshape);
                m_key_precisions.push_back(cache_precision);
                break;
            } else if (name.find("value_cache.") == 0) {
                pshape = input.get_partial_shape();
                OPENVINO_ASSERT(pshape.rank().is_static() &&
                                    pshape[1].is_static() && pshape[2].is_static() && pshape[3].is_static(),
                                "KV value cache shape must be static");
                m_block_size_in_bytes +=
                    pshape[1].get_length() * pshape[2].get_length() * pshape[3].get_length() *
                    cache_precision.size() / sub_byte_data_type_multiplier(cache_precision);
                m_value_shapes.push_back(pshape);
                m_value_precisions.push_back(cache_precision);
                break;
            }
        }
    }

    m_num_decoder_layers = m_value_precisions.size();
    OPENVINO_ASSERT(m_num_decoder_layers == m_key_precisions.size(),
                    "Invalid case: a different number of K and V caches in a LLM model");
}

ov::Shape CacheManager::set_kv_blocks(ov::PartialShape pshape, size_t num_kv_blocks) {
    pshape[0] = num_kv_blocks;
    return pshape.get_shape();
}

void CacheManager::update_request_tensor(size_t decoder_layer_id) {
    m_request.set_tensor("key_cache." + std::to_string(decoder_layer_id), m_key_cache[decoder_layer_id]);
    m_request.set_tensor("value_cache." + std::to_string(decoder_layer_id), m_value_cache[decoder_layer_id]);
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

size_t CacheManager::sub_byte_data_type_multiplier(const ov::element::Type data_type) const {
    if (data_type == ov::element::i4 || data_type == ov::element::u4)
        return 2;
    return 1;
}

void CacheManager::allocate_cache_if_needed(size_t num_kv_blocks) {
    std::unique_lock<std::shared_mutex> lk(m_mu);
    if (m_num_allocated_kv_blocks >= num_kv_blocks) {
        return;
    }

    m_num_allocated_kv_blocks = num_kv_blocks;

    ov::Coordinate start_key{0, 0, 0, 0};
    ov::Coordinate start_value{0, 0, 0, 0};

    for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
        ov::Shape key_cache_shape = set_kv_blocks(m_key_shapes[decoder_layer_id], num_kv_blocks);
        ov::Shape value_cache_shape = set_kv_blocks(m_value_shapes[decoder_layer_id], num_kv_blocks);

        ov::Tensor key_cache, value_cache;

        if (m_context) {
            key_cache = m_context.create_tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
            value_cache = m_context.create_tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);
        } else {
            key_cache = ov::Tensor(get_key_cache_precision(decoder_layer_id), key_cache_shape);
            value_cache = ov::Tensor(get_value_cache_precision(decoder_layer_id), value_cache_shape);
        }

        if (decoder_layer_id < m_key_cache.size()) {
            auto copy_roi = [&](ov::Tensor& dst, const ov::Tensor& src, const ov::Coordinate& shape) {
                if (dst.get_element_type() == ov::element::u4 || dst.get_element_type() == ov::element::i4) {
                    size_t elems = std::accumulate(shape.begin(), shape.end(), size_t{1}, std::multiplies<size_t>());
                    std::memcpy(dst.data(), src.data(),
                                (elems * dst.get_element_type().size()) /
                                    sub_byte_data_type_multiplier(dst.get_element_type()));
                } else {
                    ov::Tensor roi(dst, start_key, shape);
                    src.copy_to(roi);
                }
            };

            copy_roi(key_cache, m_key_cache[decoder_layer_id], m_key_cache[decoder_layer_id].get_shape());
            copy_roi(value_cache, m_value_cache[decoder_layer_id], m_value_cache[decoder_layer_id].get_shape());

            m_key_cache[decoder_layer_id] = key_cache;
            m_value_cache[decoder_layer_id] = value_cache;
        } else {
            m_key_cache.push_back(key_cache);
            m_value_cache.push_back(value_cache);
        }

        update_request_tensor(decoder_layer_id);
    }
}

ov::Tensor CacheManager::get_key_cache(size_t decoder_layer_id) const {
    std::shared_lock<std::shared_mutex> lk(m_mu);
    OPENVINO_ASSERT(decoder_layer_id < m_key_cache.size());
    return m_key_cache[decoder_layer_id];
}

ov::Tensor CacheManager::get_value_cache(size_t decoder_layer_id) const {
    std::shared_lock<std::shared_mutex> lk(m_mu);
    OPENVINO_ASSERT(decoder_layer_id < m_value_cache.size());
    return m_value_cache[decoder_layer_id];
}

void CacheManager::copy_blocks(const std::map<size_t, std::list<size_t>>& block_copy_map) {
    std::unique_lock<std::shared_mutex> lk(m_mu);
    for (const auto& [src_block_id, dst_block_ids] : block_copy_map) {
        for (size_t dst_block_id : dst_block_ids) {
            for (size_t decoder_layer_id = 0; decoder_layer_id < m_num_decoder_layers; ++decoder_layer_id) {
                auto copy_one_block = [&](ov::Tensor& dst, const ov::Tensor& src,
                                          size_t src_start, size_t dst_start, size_t stride_bytes) {
                    const bool is_remote = dst.is<ov::RemoteTensor>() || src.is<ov::RemoteTensor>();
                    if (!is_remote) {
                        const uint8_t* src_ptr =
                            reinterpret_cast<const uint8_t*>(src.data()) + src_start * stride_bytes;
                        uint8_t* dst_ptr =
                            reinterpret_cast<uint8_t*>(dst.data()) + dst_start * stride_bytes;
                        std::memcpy(dst_ptr, src_ptr, stride_bytes);
                    } else {
                        // ROI copy along dim-0
                        ov::Shape roi_shape = dst.get_shape();
                        roi_shape[0] = 1;
                        ov::Coordinate src_begin{src_start, 0, 0, 0};
                        ov::Coordinate dst_begin{dst_start, 0, 0, 0};
                        ov::Tensor src_roi(src, src_begin, roi_shape);
                        ov::Tensor dst_roi(dst, dst_begin, roi_shape);
                        src_roi.copy_to(dst_roi);
                    }
                };

                const auto& key_shape = m_key_cache[decoder_layer_id].get_shape();
                const auto& val_shape = m_value_cache[decoder_layer_id].get_shape();

                auto k_type = m_key_cache[decoder_layer_id].get_element_type();
                auto v_type = m_value_cache[decoder_layer_id].get_element_type();

                size_t elems_k =
                    std::accumulate(key_shape.begin() + 1, key_shape.end(), size_t{1}, std::multiplies<size_t>());
                size_t elems_v =
                    std::accumulate(val_shape.begin() + 1, val_shape.end(), size_t{1}, std::multiplies<size_t>());

                size_t stride_k = (elems_k * k_type.size()) / sub_byte_data_type_multiplier(k_type);
                size_t stride_v = (elems_v * v_type.size()) / sub_byte_data_type_multiplier(v_type);

                copy_one_block(m_key_cache[decoder_layer_id],
                               m_key_cache[decoder_layer_id],
                               src_block_id,
                               dst_block_id,
                               stride_k);
                copy_one_block(m_value_cache[decoder_layer_id],
                               m_value_cache[decoder_layer_id],
                               src_block_id,
                               dst_block_id,
                               stride_v);
            }
        }
    }
}

}  // namespace ov::template_plugin::cache
