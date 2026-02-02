// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_infer_utils.hpp"

#include <iomanip>
#include <iostream>

#include "../logging.hpp"
#include "../util.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/make_tensor.hpp"

namespace ov {
namespace npuw {
namespace moe {

// MoEProfile constructor - uses existing profiling infrastructure
MoEProfile::MoEProfile() {
    iterative.area = "MoE Expert Iterative";  // EXPERT_ITERATIVE mode profiling
    batch.area = "MoE Expert Batch";          // EXPERT_BATCH mode profiling
    iterative.report_on_die = ov::npuw::profiling_enabled();
    batch.report_on_die = ov::npuw::profiling_enabled();
}

ov::Tensor slice_expert_weight(const ov::Tensor& batched_weight, size_t expert_id, size_t num_experts) {
    // Slice weight tensor from batched (num_experts, ...) to single expert (1, ...)
    auto shape = batched_weight.get_shape();
    if (shape.empty() || shape[0] != num_experts) {
        NPUW_ASSERT(false && "Expected batched weight with first dimension equal to num_experts");
    }

    // Calculate new shape: replace first dimension with 1
    ov::Shape view_shape = shape;
    view_shape[0] = 1;

    // Check if element type is sub-byte (4-bit types)
    // For sub-byte types, we can't use strides, but we can create a zero-copy view
    // by wrapping the data pointer at the correct byte offset
    auto elem_type = batched_weight.get_element_type();
    if (elem_type == ov::element::nf4 || elem_type == ov::element::u4 || elem_type == ov::element::i4) {
        // Calculate byte-level offset for this expert
        size_t total_byte_size = batched_weight.get_byte_size();
        size_t expert_byte_size = total_byte_size / num_experts;

        // Get pointer to this expert's data
        const uint8_t* base_ptr = static_cast<const uint8_t*>(batched_weight.data());
        void* expert_ptr = const_cast<uint8_t*>(base_ptr + expert_id * expert_byte_size);

        // Create zero-copy tensor wrapping the expert's data slice
        ov::Tensor expert_tensor(elem_type, view_shape, expert_ptr);

        LOG_DEBUG("Sliced expert " << expert_id << " weight (4-bit, zero-copy): " << shape << " -> " << view_shape);
        return expert_tensor;
    }

    // For >= 8-bit types, use util::view to create zero-copy strided view
    auto view_impl = ov::npuw::util::view(ov::get_tensor_impl(batched_weight), 0, expert_id, 1);
    ov::Tensor view_tensor = ov::make_tensor(view_impl);

    LOG_DEBUG("Sliced expert " << expert_id << " weight using util::view: " << shape << " -> "
                               << view_tensor.get_shape());

    return view_tensor;
}

std::vector<size_t> parse_selected_experts_from_router(const ov::SoPtr<ov::ITensor>& router_output,
                                                       size_t num_experts,
                                                       std::map<size_t, std::vector<size_t>>& token_to_experts,
                                                       std::map<size_t, std::vector<size_t>>& expert_to_tokens) {
    if (!router_output) {
        NPUW_ASSERT(false && "Router output tensor is null");
    }

    // Clear input maps
    token_to_experts.clear();
    expert_to_tokens.clear();

    // Expected router output shape: [num_experts, 1, token_num, 1]
    auto shape = router_output->get_shape();
    if (shape.size() != 4 || shape[0] != num_experts || shape[1] != 1 || shape[3] != 1) {
        NPUW_ASSERT(false && "Unexpected router output shape!");
    }

    size_t num_tokens = shape[2];  // token_num from shape

    // Parse which expert each token selects based on non-zero weights
    auto parse_experts = [&](auto* data) {
        // For each token, find which experts have non-zero weights
        for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
            for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
                // Index calculation for shape [num_experts, 1, token_num, 1]
                // data[expert_id, 0, token_id, 0]
                size_t idx = expert_id * num_tokens + token_id;

                float value = std::abs(static_cast<float>(data[idx]));
                if (value > 1e-6f) {
                    // This token selected this expert
                    token_to_experts[token_id].push_back(expert_id);
                    expert_to_tokens[expert_id].push_back(token_id);
                }
            }
        }
    };

    auto elem_type = router_output->get_element_type();
    if (elem_type == ov::element::f32) {
        parse_experts(router_output->data<float>());
    } else if (elem_type == ov::element::f16) {
        parse_experts(router_output->data<ov::float16>());
    } else {
        NPUW_ASSERT(false && "Unsupported element type in router output tensor");
    }

    // Convert expert_to_tokens keys to vector
    std::vector<size_t> selected_experts;
    selected_experts.reserve(expert_to_tokens.size());
    for (const auto& [expert_id, tokens] : expert_to_tokens) {
        selected_experts.push_back(expert_id);
    }

    return selected_experts;
}

void set_tensor_optimized(ov::SoPtr<ov::IAsyncInferRequest> request,
                          const ov::Output<const ov::Node>& iport,
                          const ov::SoPtr<ov::ITensor>& tensor_impl) {
    // Optimization: For small tensors, use copy instead of set_tensor to avoid overhead
    // Threshold: 11520 bytes (~11.25KB, typically 5760 f16 elements or 2880 f32 elements)
    // Empirically verified to have performance benefit for small tensors
    // Typical shapes: 1x2880x1 (f16: 5760 bytes), 1x5760x1 (f16: 11520 bytes)
    constexpr size_t SMALL_TENSOR_THRESHOLD_BYTES = 11520;

    size_t tensor_bytes = tensor_impl->get_byte_size();

    if (tensor_bytes <= SMALL_TENSOR_THRESHOLD_BYTES) {
        // Small tensor: direct copy to avoid set_tensor overhead (~0.65ms per call)
        // Copy is faster for small tensors due to avoiding NPU plugin overhead
        auto clparam = request->get_tensor(iport);
        tensor_impl->copy_to(clparam._ptr);
        LOG_DEBUG("Using copy for small tensor (" << tensor_bytes << " bytes)");
    } else {
        // Large tensor: use set_tensor (zero-copy)
        request->set_tensor(iport, tensor_impl);
    }
}

void gather_router_scores(const ov::SoPtr<ov::ITensor>& router_source,
                          const ov::SoPtr<ov::ITensor>& router_dest,
                          size_t expert_id,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size) {
    auto router_source_shape = router_source->get_shape();

    // Calculate expert offset in source tensor
    size_t expert_offset;
    if (router_source_shape.size() == 4) {
        expert_offset = expert_id * router_source_shape[2];  // [num_experts, 1, token_num, 1]
    } else if (router_source_shape.size() == 2) {
        expert_offset = expert_id * router_source_shape[1];  // [num_experts, token_num]
    } else {
        NPUW_ASSERT(false && "Unexpected router source shape");
    }

    // Gather router scores for chunk tokens
    if (router_source->get_element_type() == ov::element::f16) {
        const auto* src_base = router_source->data<ov::float16>() + expert_offset;
        auto* dst_base = router_dest->data<ov::float16>();
        for (size_t i = 0; i < chunk_size; ++i) {
            dst_base[i] = src_base[token_ids[chunk_start + i]];
        }
    } else if (router_source->get_element_type() == ov::element::f32) {
        const auto* src_base = router_source->data<float>() + expert_offset;
        auto* dst_base = router_dest->data<float>();
        for (size_t i = 0; i < chunk_size; ++i) {
            dst_base[i] = src_base[token_ids[chunk_start + i]];
        }
    } else {
        NPUW_ASSERT(false && "Unsupported router element type for gathering");
    }
}

void gather_expert_inputs(const ov::SoPtr<ov::ITensor>& input_source,
                          const ov::SoPtr<ov::ITensor>& input_dest,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size) {
    auto input_shape = input_source->get_shape();

    // Determine dimensions
    size_t hidden_dim;
    size_t token_stride;
    if (input_shape.size() == 2) {
        hidden_dim = input_shape[1];
        token_stride = hidden_dim;
    } else if (input_shape.size() == 4) {
        hidden_dim = input_shape[3];
        token_stride = hidden_dim;
    } else {
        NPUW_ASSERT(false && "Unexpected expert input tensor shape");
    }

    // Gather input embeddings for chunk tokens
    if (input_source->get_element_type() == ov::element::f16) {
        const auto* src_base = input_source->data<ov::float16>();
        auto* dst_base = input_dest->data<ov::float16>();
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t token_id = token_ids[chunk_start + i];
            const auto* src_token = src_base + token_id * token_stride;
            auto* dst_token = dst_base + i * hidden_dim;
            std::memcpy(dst_token, src_token, hidden_dim * sizeof(ov::float16));
        }
    } else if (input_source->get_element_type() == ov::element::f32) {
        const auto* src_base = input_source->data<float>();
        auto* dst_base = input_dest->data<float>();
        for (size_t i = 0; i < chunk_size; ++i) {
            size_t token_id = token_ids[chunk_start + i];
            const auto* src_token = src_base + token_id * token_stride;
            auto* dst_token = dst_base + i * hidden_dim;
            std::memcpy(dst_token, src_token, hidden_dim * sizeof(float));
        }
    } else {
        NPUW_ASSERT(false && "Unsupported expert input element type for gathering");
    }
}

void scatter_expert_outputs(const ov::SoPtr<ov::ITensor>& expert_output,
                            const ov::SoPtr<ov::ITensor>& global_output_buffer,
                            const std::vector<size_t>& token_ids,
                            size_t chunk_start,
                            size_t chunk_size,
                            size_t embed_dim,
                            size_t input_token_count,
                            const std::vector<size_t>& expert_slots_for_tokens) {
    auto elem_type = global_output_buffer->get_element_type();

    for (size_t i = 0; i < chunk_size; ++i) {
        size_t original_token_id = token_ids[chunk_start + i];
        size_t expert_slot = expert_slots_for_tokens[chunk_start + i];

        // Calculate offsets
        size_t src_offset = i * embed_dim;
        size_t dst_offset = expert_slot * input_token_count * embed_dim + original_token_id * embed_dim;

        // Scatter output to global buffer
        if (elem_type == ov::element::f32) {
            const float* src = expert_output->data<float>() + src_offset;
            float* dst = global_output_buffer->data<float>() + dst_offset;
            std::memcpy(dst, src, embed_dim * sizeof(float));
        } else if (elem_type == ov::element::f16) {
            const ov::float16* src = expert_output->data<ov::float16>() + src_offset;
            ov::float16* dst = global_output_buffer->data<ov::float16>() + dst_offset;
            std::memcpy(dst, src, embed_dim * sizeof(ov::float16));
        } else {
            OPENVINO_THROW("MoE: Unsupported element type for chunk output relayout: ", elem_type);
        }
    }
}

// ====================================================================================================
// MoE Request Cache Implementation
// ====================================================================================================

RequestCache::RequestCache(size_t num_layers, size_t pool_size_per_layer) : m_pool_size_per_layer(pool_size_per_layer) {
    m_pool.resize(num_layers);
    m_cache_lookup.resize(num_layers);
    m_free_list.resize(num_layers);
    m_lru_index.resize(num_layers);

    m_report_on_die = ov::npuw::profiling_enabled();
}

RequestCache::~RequestCache() {
    print_statistics();
}

void RequestCache::initialize_layer(size_t sublayer_idx, std::vector<RqPtr>&& requests) {
    NPUW_ASSERT(requests.size() == m_pool_size_per_layer && "Request count must match pool size");
    NPUW_ASSERT(sublayer_idx < m_pool.size() && "Invalid sublayer index");

    auto& pool = m_pool[sublayer_idx];
    auto& free_list = m_free_list[sublayer_idx];

    pool.reserve(m_pool_size_per_layer);
    free_list.reserve(m_pool_size_per_layer);

    for (size_t i = 0; i < m_pool_size_per_layer; ++i) {
        PoolEntry entry;
        entry.request = std::move(requests[i]);
        entry.is_configured = false;
        entry.last_use_iter = 0;
        pool.push_back(std::move(entry));
        free_list.push_back(i);  // All entries start as free
    }

    LOG_DEBUG("MoE Cache: Initialized layer " << sublayer_idx << " with " << m_pool_size_per_layer << " pool entries");
}

RequestCache::RqPtr RequestCache::find(size_t sublayer_idx, const std::vector<size_t>& expert_ids) {
    NPUW_ASSERT(sublayer_idx < m_cache_lookup.size() && "Invalid sublayer index");

    m_total_queries++;

    auto& lookup = m_cache_lookup[sublayer_idx];
    std::string cache_key = make_cache_key(expert_ids);

    auto it = lookup.find(cache_key);
    if (it != lookup.end()) {
        // Cache HIT
        m_cache_hits++;
        size_t pool_idx = it->second;
        auto& pool = m_pool[sublayer_idx];
        auto& entry = pool[pool_idx];

        LOG_VERB("MoE Cache HIT for sublayer[" << sublayer_idx << "] experts: " << cache_key);

        // Update LRU timestamp using m_total_queries (monotonically increasing)
        auto& lru_index = m_lru_index[sublayer_idx];
        lru_index.erase({entry.last_use_iter, pool_idx});
        entry.last_use_iter = m_total_queries;
        lru_index.insert({entry.last_use_iter, pool_idx});

        return entry.request;
    }

    // Cache MISS
    LOG_VERB("MoE Cache MISS for sublayer[" << sublayer_idx << "] experts: " << cache_key);
    return {};
}

std::pair<RequestCache::RqPtr, size_t> RequestCache::get_idle_or_lru(size_t sublayer_idx) {
    NPUW_ASSERT(sublayer_idx < m_pool.size() && "Invalid sublayer index");

    auto& pool = m_pool[sublayer_idx];
    auto& free_list = m_free_list[sublayer_idx];
    auto& lru_index = m_lru_index[sublayer_idx];
    auto& lookup = m_cache_lookup[sublayer_idx];

    // Priority 1: Use free (unconfigured) request if available
    if (!free_list.empty()) {
        size_t pool_idx = free_list.back();
        free_list.pop_back();
        LOG_VERB("MoE Cache: Allocated free pool entry " << pool_idx << " for sublayer " << sublayer_idx);
        return {pool[pool_idx].request, pool_idx};
    }

    // Priority 2: Evict LRU entry
    NPUW_ASSERT(!lru_index.empty() && "LRU index should not be empty when free list is empty");

    auto lru_it = lru_index.begin();  // Oldest entry (smallest timestamp)
    size_t pool_idx = lru_it->second;
    auto& entry = pool[pool_idx];

    // Remove old cache entry
    std::string old_key = make_cache_key(entry.expert_ids);
    lookup.erase(old_key);
    lru_index.erase(lru_it);

    // Mark as unconfigured (will be reconfigured by caller)
    entry.is_configured = false;
    entry.expert_ids.clear();

    LOG_VERB("MoE Cache: Evicted LRU pool entry " << pool_idx << " (old key: " << old_key << ")");
    return {entry.request, pool_idx};
}

void RequestCache::register_request(size_t sublayer_idx, size_t pool_idx, const std::vector<size_t>& expert_ids) {
    NPUW_ASSERT(sublayer_idx < m_pool.size() && "Invalid sublayer index");
    NPUW_ASSERT(pool_idx < m_pool[sublayer_idx].size() && "Invalid pool index");

    auto& pool = m_pool[sublayer_idx];
    auto& lookup = m_cache_lookup[sublayer_idx];
    auto& lru_index = m_lru_index[sublayer_idx];

    auto& entry = pool[pool_idx];

    // Update entry state
    entry.expert_ids = expert_ids;
    entry.is_configured = true;
    // last_use_iter will be set by next find() call

    // Register in cache lookup
    std::string cache_key = make_cache_key(expert_ids);
    lookup[cache_key] = pool_idx;

    // Add to LRU index (with timestamp 0, will be updated on first use)
    lru_index.insert({entry.last_use_iter, pool_idx});

    LOG_VERB("MoE Cache: Registered pool entry " << pool_idx << " with key: " << cache_key);
}

std::pair<uint64_t, uint64_t> RequestCache::get_statistics() const {
    return {m_total_queries, m_cache_hits};
}

void RequestCache::print_statistics() const {
    if (m_total_queries > 0 && m_report_on_die) {
        double hit_rate = static_cast<double>(m_cache_hits) / m_total_queries * 100.0;
        std::cout << "[MoE Cache Statistics]" << std::endl;
        std::cout << "  Total Queries: " << m_total_queries << std::endl;
        std::cout << "  Cache Hits:    " << m_cache_hits << std::endl;
        std::cout << "  Cache Misses:  " << (m_total_queries - m_cache_hits) << std::endl;
        std::cout << "  Hit Rate:      " << std::fixed << std::setprecision(2) << hit_rate << "%" << std::endl;
    }
}

std::string RequestCache::make_cache_key(const std::vector<size_t>& expert_ids) const {
    std::string key;
    key.reserve(expert_ids.size() * 4);
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        if (i > 0)
            key += ",";
        key += std::to_string(expert_ids[i]);
    }
    return key;
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
