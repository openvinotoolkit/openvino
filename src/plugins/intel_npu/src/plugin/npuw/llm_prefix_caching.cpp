// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_prefix_caching.hpp"

#include "infer_request_utils.hpp"
#include "llm_infer_request.hpp"
#include "logging.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "util.hpp"

namespace ov {
namespace npuw {

bool KVBlock::add_block(const std::vector<uint64_t>& token_hashes, const KVData& kv_tensors) {
    // Check input validity
    if (token_hashes.empty()) {
        return false;
    }

    // Check if the block size exceeds capacity
    if (token_hashes.size() > m_block_size) {
        return false;
    }

    m_token_hashes = token_hashes;
    m_kv_data = kv_tensors;
    m_is_full = (token_hashes.size() == m_block_size);

    // Compute the block's hash value
    m_block_hash = compute_block_hash(token_hashes);

    return true;
}

void KVBlock::link_blocks(std::shared_ptr<KVBlock> prev_block) {
    prev_block->m_child_block_hashes.insert(m_block_hash);

    m_parent_block_hash = prev_block->m_block_hash;
}

void KVBlock::unlink_blocks(std::shared_ptr<KVBlock> prev_block) {
    prev_block->m_child_block_hashes.erase(m_block_hash);

    m_parent_block_hash = 0;
}

uint64_t KVBlock::compute_block_hash(const std::vector<uint64_t>& token_hashes) const {
    // Use the last token hash as the block hash, given token hash is calculated with preceding tokens
    return m_token_hashes.back();
}

void KVBlock::print_block_info(bool verbose) const {
    constexpr size_t BYTES_IN_MB = 1024 * 1024;

    LOG_VERB("Block information: ");
    LOG_VERB("  Block size: " << m_block_size);
    LOG_VERB("  Block hash: " << m_block_hash);
    LOG_VERB("  Token count: " << m_token_hashes.size());
    LOG_VERB("  Status: " << (m_is_full ? "Full" : "Not Full"));
    LOG_VERB("  Token start: " << m_token_start);

    LOG_VERB("  Children blocks: ");
    if (m_child_block_hashes.empty()) {
        LOG_VERB("    Null");
    } else {
        size_t index = 0;
        for (auto it = m_child_block_hashes.begin(); it != m_child_block_hashes.end(); ++it, ++index) {
            LOG_VERB("    hash [" << index << "]: " << *it);
        }
    }

    if (verbose) {
        LOG_VERB("  KV cache stored in block: ");
    }
    size_t total_size = 0;
    for (const auto& pair : m_kv_data) {
        const std::string& name = pair.first;
        const ov::SoPtr<ov::ITensor>& tensor = pair.second;

        total_size += tensor->get_byte_size();

        if (!verbose) {
            continue;
        }

        // Print KV cache stored in block verbosely
        LOG_VERB("Name: " << name);
        if (tensor) {
            LOG_VERB("Tensor Shape: " << tensor->get_shape().to_string());
        } else {
            LOG_VERB("Tensor is null");
        }
        LOG_VERB("----------------------------------------");
    }

    LOG_VERB("  KV cache tensor total size: " << total_size / BYTES_IN_MB << " MB");
}

bool PrefixCacheManager::put_block(const std::shared_ptr<KVBlock>& block, uint64_t prev_block_hash) {
    // Do not cache incomplete blocks
    if (!block->is_full()) {
        LOG_VERB("[Cache store] Block rejected: not full");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Check if the block is already cached
        const auto curr_block = get_block_unsafe(block->get_block_hash());
        if (curr_block != nullptr) {
            // Update LRU position if block is evictable
            update_evictable_lru_unsafe(curr_block->get_block_hash(), true);
            LOG_VERB("[Cache store] Block already cached, updated LRU");
            return true;  // Already cached counts as success
        }

        // Link current block with previous block
        const auto prev_block = get_block_unsafe(prev_block_hash);
        if (prev_block != nullptr) {
            // Link the current block with the previous block before attempting eviction.
            // When the cache is full and all preceding blocks have child dependencies (e.g., A -> B -> C -> D),
            // linking ensures no eviction candidates are available, preventing block E from being added.
            block->link_blocks(prev_block);
            // Previous block now has a child, so it's no longer evictable
            update_evictable_lru_unsafe(prev_block_hash, false);
        } else if (prev_block_hash != 0) {
            // If the previous block wasn't added due to full cache capacity,
            // there's no need to add the current block, as it won't be accessed in the cache.
            LOG_VERB("[Cache store] Block rejected: previous block missing");
            return false;
        }

        if (m_cache_map.size() >= m_max_cache_size) {
            if (!evict_lru_block_unsafe()) {
                // New block is not added into the cache
                if (prev_block != nullptr) {
                    block->unlink_blocks(prev_block);
                    // Restore previous block's evictable status if it has no other children
                    if (prev_block->get_child_block_hashes().empty()) {
                        update_evictable_lru_unsafe(prev_block_hash, true);
                    }
                }
                LOG_VERB("[Cache store] Block rejected: cache full, no eviction candidate");
                return false;
            }
        }

        m_cache_map[block->get_block_hash()] = block;
        // New added block is a leaf node - mark as evictable (automatically adds to LRU list)
        update_evictable_lru_unsafe(block->get_block_hash(), true);

        LOG_VERB("[Cache store] Successfully added block. Token start: " << block->get_token_start()
                                                                         << " block hash: " << block->get_block_hash());
    }

    return true;
}

void PrefixCacheManager::update_evictable_lru_unsafe(uint64_t block_hash, bool is_evictable) {
    auto evictable_iter_it = m_evictable_lru_iter_map.find(block_hash);
    bool currently_evictable = (evictable_iter_it != m_evictable_lru_iter_map.end());

    // Case 1: Block should not be evictable
    if (!is_evictable) {
        if (currently_evictable) {
            // Remove from evictable LRU list
            m_evictable_lru_list.erase(evictable_iter_it->second);
            m_evictable_lru_iter_map.erase(evictable_iter_it);
        }
        return;  // Nothing to do if already non-evictable
    }

    // Case 2: Block should be evictable
    if (currently_evictable) {
        // Already evictable - remove from current position to move to front
        m_evictable_lru_list.erase(evictable_iter_it->second);
    }
    // Add/move block to front (most recently used)
    m_evictable_lru_list.push_front(block_hash);
    m_evictable_lru_iter_map[block_hash] = m_evictable_lru_list.begin();
}

bool PrefixCacheManager::evict_lru_block_unsafe() {
    // O(1) eviction: directly access the least recently used evictable block
    if (m_evictable_lru_list.empty()) {
        return false;  // No evictable blocks available
    }

    // Get LRU evictable block (from back of list)
    uint64_t lru_block_hash = m_evictable_lru_list.back();

    const auto lru_block = get_block_unsafe(lru_block_hash);
    if (lru_block == nullptr) {
        // Data inconsistency - clean up and return false
        m_evictable_lru_list.pop_back();
        m_evictable_lru_iter_map.erase(lru_block_hash);
        return false;
    }

    LOG_VERB("Cache is full, evict LRU block");

    lru_block->print_block_info(false);

    // Unlink from parent block and update parent's evictable status
    const auto lru_prev_block_hash = lru_block->get_parent_block_hash();
    if (lru_prev_block_hash != 0) {
        const auto lru_prev_block = get_block_unsafe(lru_prev_block_hash);
        if (lru_prev_block != nullptr) {
            lru_block->unlink_blocks(lru_prev_block);
            // Parent might now be evictable if it has no other children
            if (lru_prev_block->get_child_block_hashes().empty()) {
                update_evictable_lru_unsafe(lru_prev_block_hash, true);
            }
        }
    }

    // Remove from cache
    m_cache_map.erase(lru_block_hash);  // O(1)

    // Remove from evictable LRU list
    m_evictable_lru_list.pop_back();                 // O(1) - we know it's at the back
    m_evictable_lru_iter_map.erase(lru_block_hash);  // O(1)

    return true;  // Successfully evicted one block
}

std::shared_ptr<KVBlock> PrefixCacheManager::get_block(uint64_t combined_hash) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto block = get_block_unsafe(combined_hash);
    if (block != nullptr) {
        // Update LRU position if block is evictable
        update_evictable_lru_unsafe(block->get_block_hash(), true);
    }
    return block;
}

std::shared_ptr<KVBlock> PrefixCacheManager::get_block_unsafe(uint64_t combined_hash) const {
    auto it = m_cache_map.find(combined_hash);
    if (it != m_cache_map.end()) {
        return it->second;
    }

    return nullptr;
}

void PrefixCacheManager::print_cache_status(bool verbose) const {
    LOG_VERB("Cache Status:");
    LOG_VERB("Max Cache Size: " << m_max_cache_size);
    LOG_VERB("Number of Cached Blocks: " << m_cache_map.size());
    LOG_VERB("----------------------------------------");

    // Print information of all blocks in cache
    for (const auto& pair : m_cache_map) {
        uint64_t key = pair.first;
        std::shared_ptr<KVBlock> block = pair.second;

        LOG_VERB("Key Hash: " << key);
        if (block) {
            block->print_block_info(verbose);
        } else {
            LOG_VERB("  Block is null");
        }
        LOG_VERB("----------------------------------------");
    }
}

// ============================================================================
// PrefixCachingHelper Implementation
// ============================================================================

PrefixCachingHelper::PrefixCachingHelper(LLMInferRequest& request)
    : m_request(request),
      m_cache_manager(
          std::make_shared<PrefixCacheManager>(request.m_npuw_llm_compiled_model->m_prefix_caching_max_num_blocks)) {}

PrefixCacheRestorationContext PrefixCachingHelper::prepare_and_restore(const ov::SoPtr<ov::ITensor>& input_ids,
                                                                       uint64_t input_prompt_len) {
    PrefixCacheRestorationContext context;

    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;

    // Calculate input prompts hash
    context.prompt_hashes = calculate_hashes(input_ids);

    // Try to restore prefilled prompts from cache
    context.restored_token_num = restore_blocks(input_ids, context.prompt_hashes);

    uint64_t scheduled_token_num = input_prompt_len - context.restored_token_num;
    LOG_VERB("[PrefixCache] Successfully restored " << context.restored_token_num
                                                    << " tokens from cache. "
                                                       "Will compute "
                                                    << scheduled_token_num << " tokens out of total input length "
                                                    << input_prompt_len << ".");

    context.remaining_prompts = scheduled_token_num;
    context.token_idx = context.restored_token_num;
    context.restore_prefix_cache = (scheduled_token_num < input_prompt_len);

    // Update kvcache state
    kvcache_desc.num_stored_tokens = static_cast<uint32_t>(context.restored_token_num);

    return context;
}

void PrefixCachingHelper::store_computed_blocks(size_t chunk_size,
                                                const std::vector<uint64_t>& prompt_hashes,
                                                size_t& token_idx) {
    store_blocks(chunk_size, prompt_hashes, token_idx);
}

void PrefixCachingHelper::print_cache_status(bool verbose) const {
    if (m_cache_manager) {
        m_cache_manager->print_cache_status(verbose);
    }
}

void PrefixCachingHelper::populate_attention_mask_for_restored_cache(const ov::SoPtr<ov::ITensor>& attention_mask,
                                                                     const ov::SoPtr<ov::ITensor>& attn_mask_in_tensor,
                                                                     size_t num_restored_tokens) {
    // Populate the attention mask for prefix caching:
    // num_restored_tokens have been prefilled already from cache
    // The calculated key/values blocks will be copied from cache to past k/v inputs for inference
    std::copy_n(attention_mask->data<int64_t>(), num_restored_tokens, attn_mask_in_tensor->data<int64_t>());
}

std::vector<uint64_t> PrefixCachingHelper::calculate_hashes(const ov::SoPtr<ov::ITensor>& input_ids) {
    // For LLM, model accepts 2d inputs_embeds[BATCH, SEQ_LEN]
    // For VLM, model accepts 3d inputs_ids[BATCH, SEQ_LEN, EMB_SIZE]
    bool is_input_embeds = input_ids->get_shape().size() == 2 ? false : true;

    const auto data_elem_size = input_ids->get_element_type().size();
    size_t total_size = input_ids->get_shape()[LLMInferRequest::layer_ids::INPUT_IDS_SEQ_LEN_DIM];

    std::vector<uint64_t> prompt_hashes(total_size);

    uint64_t prefix_hash = 0;
    for (size_t i = 0; i < total_size; ++i) {
        size_t offset = i * data_elem_size;
        size_t size = data_elem_size;
        if (is_input_embeds) {
            offset *= input_ids->get_shape().back();
            size *= input_ids->get_shape().back();
        }
        const char* token_data = reinterpret_cast<const char*>(input_ids->data()) + offset;
        uint64_t token_hash = std::hash<std::string_view>{}(std::string_view(token_data, size));
        prefix_hash = prefix_hash * 31 + token_hash;
        prompt_hashes[i] = prefix_hash;
    }

    return prompt_hashes;
}

std::vector<std::shared_ptr<KVBlock>> PrefixCachingHelper::find_cached_blocks(
    const ov::SoPtr<ov::ITensor>& input_ids,
    const std::vector<uint64_t>& prompt_hashes) {
    const uint64_t block_size = m_request.m_npuw_llm_compiled_model->m_prefix_caching_block_size;
    const uint64_t chunk_size = m_request.m_npuw_llm_compiled_model->m_prefill_chunk_size;
    auto& kvcache_desc = m_request.m_npuw_llm_compiled_model->m_kvcache_desc;

    size_t actual_token_num = input_ids->get_shape()[LLMInferRequest::layer_ids::INPUT_IDS_SEQ_LEN_DIM];
    size_t num_blocks = (actual_token_num + block_size - 1) / block_size;

    uint64_t max_restored_token_num =
        kvcache_desc.max_prompt_size - m_request.m_npuw_llm_compiled_model->m_prefill_chunk_size;

    std::vector<std::shared_ptr<KVBlock>> cached_blocks;
    size_t token_idx = 0;
    uint64_t restored_token_num = 0;

    for (size_t block_index = 0; block_index < num_blocks; ++block_index) {
        if ((actual_token_num - block_index * block_size) < block_size) {
            break;  // Not a full block
        }

        std::vector<uint64_t> token_hashes(block_size);
        for (size_t i = 0; i < block_size; ++i) {
            token_hashes[i] = prompt_hashes[token_idx];
            token_idx++;
        }

        uint64_t block_hash = token_hashes.back();

        auto retrieved_block = m_cache_manager->get_block(block_hash);
        if (!retrieved_block) {
            LOG_VERB("[PrefixCache] No cache block found for hash " << block_hash);
            break;
        }

        LOG_VERB("[PrefixCache] Cache hit for block hash " << block_hash << ", tokens start from position "
                                                           << retrieved_block->get_token_start());

        cached_blocks.push_back(retrieved_block);
        restored_token_num += block_size;

        // Ensure the cached tokens can be loaded into infer request
        if (restored_token_num + block_size > max_restored_token_num) {
            break;
        }

        // At least we should infer "1" token to generate "logit"
        if (restored_token_num == actual_token_num) {
            cached_blocks.pop_back();  // Remove the last block
            break;
        }
    }

    // Align restored token count down to chunk size boundary for consistency
    // This prevents KV cache mismatches that can occur when partial chunks use different padding strategies
    // between cache storage and runtime inference, ensuring exact numerical equivalence
    if (restored_token_num > 0 && restored_token_num % chunk_size != 0) {
        uint64_t aligned_token_num = (restored_token_num / chunk_size) * chunk_size;
        uint64_t blocks_to_remove = (restored_token_num - aligned_token_num) / block_size;

        LOG_VERB("[PrefixCache] Aligning restored token count from "
                 << restored_token_num << " down to " << aligned_token_num << " (chunk size boundary), removing "
                 << blocks_to_remove << " blocks");

        // Remove the excess blocks that don't align with chunk boundary
        for (uint64_t i = 0; i < blocks_to_remove; ++i) {
            cached_blocks.pop_back();
        }
    }

    return cached_blocks;
}

uint64_t PrefixCachingHelper::restore_blocks(const ov::SoPtr<ov::ITensor>& input_ids,
                                             const std::vector<uint64_t>& prompt_hashes) {
    // Find all cached blocks first
    auto cached_blocks = find_cached_blocks(input_ids, prompt_hashes);

    // Both Block KV and Continuous KV modes handle restoration inside their own
    // apply_cached_prefix_blocks() override:
    //   Block KV:    injects cached NPU tensors into KVCacheBlockManagers (zero-copy).
    //   Continuous:  copies CPU KV tensors into the prefill past_key_values input ports.
    m_request.m_kvcache_strategy->apply_cached_prefix_blocks(cached_blocks);

    // Calculate total restored token number
    const uint64_t block_size = m_request.m_npuw_llm_compiled_model->m_prefix_caching_block_size;
    uint64_t restored_token_num = cached_blocks.size() * block_size;

    LOG_VERB("[PrefixCache] Successfully found " << cached_blocks.size() << " cached blocks, "
                                                 << "restored " << restored_token_num << " tokens");

    return restored_token_num;
}

void PrefixCachingHelper::store_blocks(size_t chunk_size,
                                       const std::vector<uint64_t>& prompt_hashes,
                                       size_t& token_idx) {
    const uint64_t block_size = m_request.m_npuw_llm_compiled_model->m_prefix_caching_block_size;

    if (chunk_size < block_size) {
        return;
    }

    // Skip caching for partial chunks to maintain KV cache consistency
    // Partial chunks involve padding operations during inference that can introduce numerical variations
    // Caching only full chunks ensures deterministic behavior and avoids subtle differences
    if (chunk_size < m_request.m_npuw_llm_compiled_model->m_prefill_chunk_size) {
        return;
    }

    const uint64_t chunk_prompt_len = m_request.m_npuw_llm_compiled_model->m_prefill_chunk_size;
    // pad_offset: number of padding slots prepended when this chunk is shorter than chunk_prompt_len.
    // The valid token KV data in the padded output buffer starts at pad_offset.
    const size_t pad_offset = chunk_size < chunk_prompt_len ? chunk_prompt_len - chunk_size : 0;

    for (size_t out_token_offset = pad_offset; out_token_offset < chunk_prompt_len; out_token_offset += block_size) {
        if ((chunk_prompt_len - out_token_offset) < block_size) {
            break;  // Not a full block
        }

        // Collect token hashes for this block
        std::vector<uint64_t> token_hashes(block_size);
        for (size_t i = 0; i < block_size; ++i) {
            token_hashes[i] = prompt_hashes[token_idx];
            token_idx++;
        }
        const size_t block_token_start = token_idx - block_size;

        // Delegate block creation entirely to the active strategy:
        //   Block KV:    captures SoPtr refs to the NPU block tensors (zero-copy).
        //   Continuous:  CPU-copies the output slice [out_token_offset, out_token_offset+block_size) into
        //                a fresh KVBlock.
        auto block = m_request.m_kvcache_strategy->make_prefix_block(block_token_start,
                                                                     out_token_offset,
                                                                     block_size,
                                                                     token_hashes);
        OPENVINO_ASSERT(block, "make_prefix_block returned nullptr — strategy must produce a valid block");

        uint64_t prev_block_hash = 0;
        if (block->get_token_start() > 0) {
            prev_block_hash = prompt_hashes[block->get_token_start() - 1];
        }

        // Note: put_block returns bool but we don't need to check it here.
        // The block's lifetime is managed by shared_ptr reference counting.
        // If cache accepts it, both cache and this function hold references.
        // If cache rejects it, only this function holds reference and block will be destroyed
        // when it goes out of scope, which is the expected behavior.
        m_cache_manager->put_block(block, prev_block_hash);
    }
}

}  // namespace npuw
}  // namespace ov
