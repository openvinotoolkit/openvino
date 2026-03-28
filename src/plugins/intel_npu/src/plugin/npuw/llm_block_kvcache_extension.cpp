// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_block_kvcache_extension.hpp"

#include <regex>

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_base_request.hpp"
#include "logging.hpp"
#include "util.hpp"

namespace {

// Construct numbered block input name: e.g. "past_key_values.5.key_block_2"
std::string make_numbered_block_input_name(const std::string& kv_type, const std::string& layer_idx, size_t block_idx) {
    return "past_key_values." + layer_idx + "." + kv_type + "_block_" + std::to_string(block_idx);
}

// Construct block_tail input name: e.g. "past_key_values.5.key_block_tail"
std::string make_block_tail_input_name(const std::string& kv_type, const std::string& layer_idx) {
    return "past_key_values." + layer_idx + "." + kv_type + "_block_tail";
}

// Set dummy tensors on all numbered block inputs in a ports map.
// Returns the count of block inputs that were set.
template <typename PortMapType, typename SetTensorFn>
size_t set_dummy_block_tensors(const PortMapType& ports_map,
                               SetTensorFn&& set_tensor_fn,
                               const ov::SoPtr<ov::ITensor>& dummy_key_tensor,
                               const ov::SoPtr<ov::ITensor>& dummy_value_tensor) {
    size_t block_count = 0;
    for (const auto& [name, port] : ports_map) {
        if (name.find("block_tail") != std::string::npos) {
            continue;
        }
        if (name.find("_block_") != std::string::npos) {
            bool is_key_block = (name.find("key_block") != std::string::npos);
            set_tensor_fn(port, is_key_block ? dummy_key_tensor : dummy_value_tensor);
            block_count++;
        }
    }
    return block_count;
}

// Pre-categorised block ports for one layer, ready for BlockBindingHelper::from_ports().
struct LayerBlockPorts {
    std::vector<ov::Output<const ov::Node>> numbered;  // block_0, block_1, …
    std::optional<ov::Output<const ov::Node>> tail;    // block_tail (nullopt if absent)
};

// Partition block input ports for one layer into numbered (block_0, block_1, …) and tail.
LayerBlockPorts partition_layer_block_ports(
    const std::string& kv_type,
    const std::string& layer_idx,
    const std::unordered_map<std::string, ov::Output<const ov::Node>>& ports_map,
    uint32_t max_blocks) {
    LayerBlockPorts result;
    result.numbered.reserve(max_blocks);

    for (uint32_t idx = 0; idx < max_blocks; ++idx) {
        auto it = ports_map.find(make_numbered_block_input_name(kv_type, layer_idx, static_cast<size_t>(idx)));
        if (it != ports_map.end()) {
            result.numbered.push_back(it->second);
        } else {
            break;  // This variant has fewer blocks than max_blocks
        }
    }

    auto tail_it = ports_map.find(make_block_tail_input_name(kv_type, layer_idx));
    if (tail_it != ports_map.end()) {
        result.tail = tail_it->second;
    }

    return result;
}

// Copy a block tensor (or a slice of it) into the tail input port, handling padding correctly.
void copy_block_to_tail_input(const ov::SoPtr<ov::ITensor>& block_tensor,
                              uint32_t start_token,
                              uint32_t end_token,
                              uint32_t kv_dim,
                              const ov::Output<const ov::Node>& tail_input_port,
                              std::shared_ptr<ov::IAsyncInferRequest> request) {
    namespace uu = ov::npuw::util;
    NPUW_ASSERT(start_token <= end_token);

    auto tail_input_tensor = request->get_tensor(tail_input_port);
    NPUW_ASSERT(end_token <= static_cast<uint32_t>(tail_input_tensor->get_shape()[kv_dim]));

    auto src_slice = uu::make_tensor_slice(block_tensor, kv_dim, start_token, end_token);
    auto dst_slice = uu::make_tensor_slice(tail_input_tensor, kv_dim, start_token, end_token);
    uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
}

}  // anonymous namespace

// ============================================================================
// BlockKVCacheExtension — public methods
// ============================================================================

namespace ov {
namespace npuw {

bool BlockKVCacheExtension::initialize(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
    const PortsMap& prefill_in_ports,
    const PortsMap& prefill_out_ports,
    const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports,
    const std::string& device,
    const std::shared_ptr<LLMCompiledModel>& compiled_model) {
    m_compiled_model = compiled_model;
    m_device = device;

    if (!m_compiled_model->m_cfg.get<::intel_npu::NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE>()) {
        return false;
    }

    LOG_INFO("=== Initializing Block-based KV Cache Managers ===");

    // Phase 2: Dummy tensor optimization — share dummy tensors across all block inputs
    auto block_shapes = find_block_shapes(prefill_in_ports);
    if (block_shapes.found_key) {
        auto dummy_tensors = allocate_dummy_block_tensors(block_shapes);
        set_dummy_tensors_to_all_requests(dummy_tensors,
                                          prefill_request,
                                          generate_requests,
                                          prefill_in_ports,
                                          gen_variant_in_ports);
    }

    // Phase 3: Parse structure
    auto layer_blocks = parse_block_inputs_structure(prefill_in_ports);

    // Phase 4: Create managers and pre-compute binding helpers
    create_block_managers_and_helpers(layer_blocks, prefill_in_ports, generate_requests, gen_variant_in_ports);

    // Phase 5: Snapshot original prefill output tensors for restore_prefill_output_buffers().
    // These are the model-owned buffers that exist before any zero-copy redirect happens.
    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const std::string layer_str = std::to_string(layer_idx);
        for (const char* kv_type : {"key", "value"}) {
            std::string output_name = "present." + layer_str + "." + kv_type;
            auto port_it = prefill_out_ports.find(output_name);
            if (port_it != prefill_out_ports.end()) {
                m_prefill_original_output_tensors[output_name] = prefill_request->get_tensor(port_it->second);
            }
        }
    }
    LOG_INFO("Snapshotted " << m_prefill_original_output_tensors.size() << " original prefill output tensors");

    LOG_INFO("=== Block-based KV Cache Initialization Complete ===");

    m_enabled = true;
    return true;
}

uint32_t BlockKVCacheExtension::get_block_size() const {
    OPENVINO_ASSERT(!m_kv_cache_block_managers.empty());
    return m_kv_cache_block_managers.begin()->second.key_manager->get_block_size();
}

void BlockKVCacheExtension::reset() {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }
    for (auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        layer_managers.key_manager->clear_all();
        layer_managers.value_manager->clear_all();
    }
}

// ============================================================================
// Prefill path
// ============================================================================

void BlockKVCacheExtension::load_past_kv_blocks_to_prefill(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const PortsMap& prefill_in_ports) {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    LOG_DEBUG("=== Binding Block Tensors to Prefill Model Inputs ===");

    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const std::string layer_idx_str = std::to_string(layer_idx);

        auto bind_kv_blocks = [&](KVCacheBlockManager* manager, const char* kv_type) {
            if (!manager) {
                return;
            }
            auto allocated_blocks = manager->get_allocated_blocks();
            for (size_t block_idx = 0; block_idx < allocated_blocks.size(); ++block_idx) {
                std::string input_name = make_numbered_block_input_name(kv_type, layer_idx_str, block_idx);
                auto port_it = prefill_in_ports.find(input_name);
                if (port_it != prefill_in_ports.end()) {
                    prefill_request->set_tensor(port_it->second,
                                                manager->get_block_tensor(allocated_blocks[block_idx]));
                    LOG_VERB("Bound " << kv_type << " block layer " << layer_idx_str << " block_" << block_idx);
                }
            }
        };

        bind_kv_blocks(layer_managers.key_manager.get(), "key");
        bind_kv_blocks(layer_managers.value_manager.get(), "value");
    }
}

bool BlockKVCacheExtension::redirect_prefill_outputs_to_new_blocks(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const PortsMap& prefill_out_ports,
    uint32_t num_new_tokens) {
    if (m_kv_cache_block_managers.empty()) {
        return false;
    }

    // Zero-copy optimisation: bind prefill output ports directly to fresh blocks BEFORE inference.
    // This lets the model write KV cache straight into BlockManager memory with no post-inference copy.
    //
    // Steps:
    //   1. Pre-allocate one block per layer to hold the new tokens.
    //   2. Redirect that output port to the block tensor (present.N.key / present.N.value).
    //   3. Update block metadata immediately (token count is known before inference).
    //   After infer(), data and metadata are already in place — no additional work needed.
    //
    // CONSTRAINT: Only valid when the chunk writes into exactly ONE block:
    //   - current_position must be block-aligned (start of a block boundary).
    //   - num_new_tokens must not span across a block boundary.
    // When these constraints are not met, callers must call restore_prefill_output_buffers() instead
    // and fall back to the copy-based copy_prefill_outputs_to_blocks() path.

    auto& kvcache_desc = m_compiled_model->m_kvcache_desc;
    uint32_t current_position = kvcache_desc.num_stored_tokens;

    // Verify block-alignment (caller should have checked, but assert for safety)
    uint32_t first_block_size = m_kv_cache_block_managers.begin()->second.key_manager->get_block_size();
    if (current_position % first_block_size != 0) {
        return false;
    }

    LOG_DEBUG("=== Redirecting Prefill Outputs to New Blocks (Zero-Copy) ===");
    LOG_DEBUG("Pre-allocating blocks for " << num_new_tokens << " new tokens");

    size_t total_blocks_allocated = 0;
    size_t total_outputs_bound = 0;

    auto process_kv_blocks = [&](uint32_t layer_idx, KVCacheBlockManager* manager, const char* kv_type_name) -> bool {
        uint32_t block_size = manager->get_block_size();
        uint32_t start_pos = current_position;
        uint32_t end_pos = current_position + num_new_tokens;
        uint32_t start_block_idx = start_pos / block_size;
        uint32_t end_block_idx = (end_pos - 1) / block_size;

        // Alignment postconditions for zero-copy
        OPENVINO_ASSERT(start_pos % block_size == 0,
                        "Zero-copy prefill requires block-aligned position. ",
                        "current_position=",
                        start_pos,
                        ", block_size=",
                        block_size);
        OPENVINO_ASSERT(start_block_idx == end_block_idx,
                        "Zero-copy prefill requires writing to a single block. ",
                        "start_block=",
                        start_block_idx,
                        ", end_block=",
                        end_block_idx,
                        ", num_new_tokens=",
                        num_new_tokens,
                        ", block_size=",
                        block_size);

        uint32_t tokens_in_block = end_pos - (start_block_idx * block_size);
        OPENVINO_ASSERT(tokens_in_block == block_size || tokens_in_block == num_new_tokens,
                        "Zero-copy prefill requires writing full blocks. ",
                        "tokens_in_block=",
                        tokens_in_block,
                        ", block_size=",
                        block_size,
                        ", num_new_tokens=",
                        num_new_tokens);

        // Ensure the target block is allocated
        uint32_t blocks_needed = end_block_idx + 1;
        auto allocated_blocks = manager->get_allocated_blocks();
        while (allocated_blocks.size() < blocks_needed) {
            auto new_block_id = manager->allocate_block();
            OPENVINO_ASSERT(new_block_id.has_value(),
                            "Failed to allocate ",
                            kv_type_name,
                            " block for layer ",
                            layer_idx);
            total_blocks_allocated++;
            allocated_blocks = manager->get_allocated_blocks();
        }

        std::string output_name = "present." + std::to_string(layer_idx) + "." + kv_type_name;
        auto port_it = prefill_out_ports.find(output_name);
        if (port_it != prefill_out_ports.end()) {
            uint32_t target_block_id = allocated_blocks[start_block_idx];
            auto target_block_tensor = manager->get_block_tensor(target_block_id);
            prefill_request->set_tensor(port_it->second, target_block_tensor);

            // Update metadata immediately — we know how many tokens will be written
            uint32_t block_start_pos = start_block_idx * block_size;
            manager->update_block_tokens(target_block_id, end_pos - block_start_pos);
            total_outputs_bound++;
            LOG_VERB("Bound " << kv_type_name << " output layer " << layer_idx << " to block_" << start_block_idx
                              << " (zero-copy, " << num_new_tokens << " tokens, metadata updated)");
        }
        return true;
    };

    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        if (layer_managers.key_manager) {
            process_kv_blocks(layer_idx, layer_managers.key_manager.get(), "key");
        }
        if (layer_managers.value_manager) {
            process_kv_blocks(layer_idx, layer_managers.value_manager.get(), "value");
        }
    }

    LOG_DEBUG("Pre-binding complete: allocated=" << total_blocks_allocated << " blocks, bound=" << total_outputs_bound
                                                 << " outputs");
    m_prefill_outputs_redirected = true;
    return true;
}

void BlockKVCacheExtension::restore_prefill_output_buffers(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const PortsMap& prefill_out_ports) {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    // CRITICAL: Must be called before any non-aligned prefill chunk to prevent silent block corruption.
    //
    // Failure scenario without this restore:
    //   Chunk 1: zero-copy binds "present.0.key" output → block 0 (block-aligned, works correctly).
    //   Chunk 2: chunk is NOT block-aligned, so zero-copy is skipped.
    //   But "present.0.key" is STILL bound to block 0 from the previous chunk.
    //   infer() now overwrites block 0 with misaligned data → silent data corruption.
    //
    // Fix: restore the original model output buffers (snapshotted in initialize()).
    // The model then writes into its own buffers, and copy_prefill_outputs_to_blocks() copies
    // the result into the correct blocks afterwards.

    if (!m_prefill_outputs_redirected) {
        LOG_VERB("Prefill outputs already pointing to original tensors - no restore needed");
        return;
    }
    OPENVINO_ASSERT(!m_prefill_original_output_tensors.empty(),
                    "Original output tensors were not snapshotted during initialize().");

    LOG_DEBUG("=== Restoring Original Output Tensors (Non-Zero-Copy Path) ===");

    size_t total_restored = 0;
    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const std::string layer_str = std::to_string(layer_idx);
        for (const char* kv_type : {"key", "value"}) {
            std::string output_name = "present." + layer_str + "." + kv_type;
            auto orig_it = m_prefill_original_output_tensors.find(output_name);
            auto port_it = prefill_out_ports.find(output_name);
            if (orig_it == m_prefill_original_output_tensors.end() || port_it == prefill_out_ports.end()) {
                continue;
            }
            auto current_tensor = prefill_request->get_tensor(port_it->second);
            if (current_tensor->data() != orig_it->second->data()) {
                prefill_request->set_tensor(port_it->second, orig_it->second);
                total_restored++;
                LOG_VERB("Restored original buffer for " << output_name);
            }
        }
    }

    LOG_DEBUG("Restore complete: restored=" << total_restored << " tensors");
    m_prefill_outputs_redirected = false;
}

// ============================================================================
// Generate path
// ============================================================================

void BlockKVCacheExtension::init_generate_kv_block_bindings(
    const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
    const PortsMap& kvcache_in_ports,
    uint32_t num_stored_tokens) {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    // Initial block binding strategy (called once per conversation on the first generate step):
    //
    //  Numbered blocks (indices within BlockBindingHelper::non_tail_capacity):
    //    → Zero-copy: set_tensor() shares the BlockManager buffer with the model input.
    //      Future writes by copy_outputs_to_blocks() go directly into the shared tensor,
    //      so subsequent generate steps need no re-binding for these blocks.
    //
    //  Tail block (index >= non_tail_capacity, handled via a separate "_block_tail" port):
    //    → Copy-based: block data is copied into the tail input with padding.
    //      Must be explicitly refreshed by update_generate_bindings() after every infer().

    LOG_DEBUG("=== Initial Block Binding for Generate Model ===");
    LOG_BLOCK();

    auto& kvcache_desc = m_compiled_model->m_kvcache_desc;
    const auto& variant_layer_helpers = m_variant_block_binding_helpers.at(kvcache_request);

    size_t total_numbered_bound = 0;
    size_t total_tail_copied = 0;

    auto process_blocks = [&](uint32_t layer_idx,
                              KVCacheBlockManager* manager,
                              const BlockBindingHelper& helper,
                              uint32_t kv_dim,
                              const char* kv_type_name) {
        auto allocated_blocks = manager->get_allocated_blocks();
        if (allocated_blocks.empty()) {
            return;
        }
        for (size_t block_idx = 0; block_idx < allocated_blocks.size(); ++block_idx) {
            uint32_t block_id = allocated_blocks[block_idx];
            auto block_tensor = manager->get_block_tensor(block_id);
            uint32_t tokens_in_block = manager->get_block_tokens(block_id);
            bool is_tail = helper.should_treat_as_tail(static_cast<uint32_t>(block_idx));
            if (is_tail) {
                copy_block_to_tail_input(block_tensor,
                                         0u,
                                         tokens_in_block,
                                         kv_dim,
                                         helper.tail_input_port.value(),
                                         kvcache_request);
                total_tail_copied++;
                LOG_VERB("Bound tail " << kv_type_name << " layer_" << layer_idx << " block_" << block_idx);
            } else {
                if (block_idx < helper.numbered_input_ports.size()) {
                    kvcache_request->set_tensor(helper.numbered_input_ports[block_idx], block_tensor);
                    total_numbered_bound++;
                    LOG_VERB("Bound numbered " << kv_type_name << " layer_" << layer_idx << " block_" << block_idx);
                }
            }
        }
    };

    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const auto& layer_helpers = variant_layer_helpers.at(layer_idx);
        if (layer_managers.key_manager) {
            process_blocks(layer_idx,
                           layer_managers.key_manager.get(),
                           layer_helpers.key_helper,
                           kvcache_desc.dim,
                           "key");
        }
        if (layer_managers.value_manager) {
            const uint32_t kv_dim = kvcache_desc.v_tensors_transposed_gen ? 3u : kvcache_desc.dim;
            process_blocks(layer_idx, layer_managers.value_manager.get(), layer_helpers.value_helper, kv_dim, "value");
        }
    }

    LOG_DEBUG("Initial binding complete: numbered=" << total_numbered_bound << ", tail=" << total_tail_copied);
}

void BlockKVCacheExtension::update_generate_bindings(uint32_t old_num_tokens,
                                                     uint32_t new_num_tokens,
                                                     const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
                                                     const PortsMap& kvcache_in_ports) {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    LOG_DEBUG("=== Update Block Bindings After Generate ===");
    LOG_BLOCK();

    auto& kvcache_desc = m_compiled_model->m_kvcache_desc;
    const auto& variant_layer_helpers = m_variant_block_binding_helpers.at(kvcache_request);

    size_t total_tail_updates = 0;
    size_t total_new_bindings = 0;

    auto update_blocks = [&](uint32_t layer_idx,
                             KVCacheBlockManager* manager,
                             const BlockBindingHelper& helper,
                             uint32_t kv_dim,
                             const char* kv_type_name) {
        auto allocated_blocks = manager->get_allocated_blocks();
        if (allocated_blocks.empty()) {
            return;
        }

        // update_generate_bindings is only called after at least one prefill has run,
        // so old_num_tokens must be positive.
        NPUW_ASSERT(old_num_tokens > 0);

        uint32_t final_block_idx = helper.get_block_index_for_position(new_num_tokens - 1);
        uint32_t old_block_idx = helper.get_block_index_for_position(old_num_tokens - 1);

        // Bind every block that was newly entered during this step (boundary crossings).
        // In standard decoding (input_tokens_len=1) this loop runs 0 or 1 times.
        // In speculative decoding it may run more times if input_tokens_len > block_size.
        for (uint32_t bidx = old_block_idx + 1; bidx <= final_block_idx; ++bidx) {
            if (bidx >= static_cast<uint32_t>(allocated_blocks.size())) {
                break;
            }
            bool block_is_tail = helper.should_treat_as_tail(bidx);
            if (block_is_tail) {
                uint32_t block_id = allocated_blocks[bidx];
                copy_block_to_tail_input(manager->get_block_tensor(block_id),
                                         0u,
                                         manager->get_block_tokens(block_id),
                                         kv_dim,
                                         helper.tail_input_port.value(),
                                         kvcache_request);
                total_tail_updates++;
                LOG_VERB("Updated tail " << kv_type_name << " layer_" << layer_idx << " block_" << bidx
                                         << " (crossed block boundary)");
            } else if (bidx < helper.numbered_input_ports.size()) {
                uint32_t block_id = allocated_blocks[bidx];
                kvcache_request->set_tensor(helper.numbered_input_ports[bidx], manager->get_block_tensor(block_id));
                total_new_bindings++;
                LOG_VERB("Bound new numbered " << kv_type_name << " layer_" << layer_idx << " block_" << bidx);
            }
        }

        // If the final block is a tail block and was NOT newly entered by the loop above
        // (i.e., old and new tokens are in the same block), incrementally copy only the
        // tokens added this step. Numbered blocks share memory with BlockManager and need
        // no explicit update.
        bool final_is_tail = helper.should_treat_as_tail(final_block_idx);
        if (final_is_tail && final_block_idx == old_block_idx) {
            uint32_t block_id = allocated_blocks[final_block_idx];
            uint32_t block_start_pos = final_block_idx * helper.block_size;
            uint32_t prev_tokens_in_block = old_num_tokens - block_start_pos;
            uint32_t new_tokens_in_block = manager->get_block_tokens(block_id);
            copy_block_to_tail_input(manager->get_block_tensor(block_id),
                                     prev_tokens_in_block,
                                     new_tokens_in_block,
                                     kv_dim,
                                     helper.tail_input_port.value(),
                                     kvcache_request);
            total_tail_updates++;
            LOG_VERB("Refreshed tail " << kv_type_name << " layer_" << layer_idx << " block_" << final_block_idx << " ["
                                       << prev_tokens_in_block << ".." << new_tokens_in_block << "]");
        }
    };

    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const auto& layer_helpers = variant_layer_helpers.at(layer_idx);
        if (layer_managers.key_manager) {
            update_blocks(layer_idx,
                          layer_managers.key_manager.get(),
                          layer_helpers.key_helper,
                          kvcache_desc.dim,
                          "key");
        }
        if (layer_managers.value_manager) {
            const uint32_t kv_dim = kvcache_desc.v_tensors_transposed_gen ? 3u : kvcache_desc.dim;
            update_blocks(layer_idx, layer_managers.value_manager.get(), layer_helpers.value_helper, kv_dim, "value");
        }
    }

    LOG_DEBUG("Update complete: tail_updates=" << total_tail_updates << ", new_bindings=" << total_new_bindings);
}

// ============================================================================
// KV copy engine — shared by prefill (copy-path) and generate
// ============================================================================

void BlockKVCacheExtension::copy_outputs_to_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                                   const PortsMap& src_ports,
                                                   uint32_t num_tokens,
                                                   bool v_transposed,
                                                   uint32_t current_kv_position) {
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();

    for (std::size_t i = LLMInferBaseRequest::layer_ids::kStartOutputKVCacheLayers; i < compiled->outputs().size();
         ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();

        bool is_key = (output_name.find("key") != std::string::npos);
        bool is_value = (output_name.find("value") != std::string::npos);
        if (!is_key && !is_value) {
            continue;
        }

        // Parse layer index from output name (e.g., "present.0.key" → layer 0)
        std::regex layer_regex(R"(present\.(\d+)\.)");
        std::smatch match;
        uint32_t layer_idx = 0;
        if (std::regex_search(output_name, match, layer_regex) && match.size() > 1) {
            layer_idx = static_cast<uint32_t>(std::stoi(match[1].str()));
        }

        auto it = m_kv_cache_block_managers.find(layer_idx);
        if (it == m_kv_cache_block_managers.end()) {
            continue;
        }

        auto& layer_managers = it->second;
        auto& block_manager = is_key ? layer_managers.key_manager : layer_managers.value_manager;

        const uint32_t kv_dim = (is_value && v_transposed) ? 3u : kvcache_desc.dim;

        auto src_tensor = request->get_tensor(src_ports.at(output_name));
        uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len, "num_tokens (", num_tokens, ") > src_seq_len (", src_seq_len, ")");

        auto src_to_copy = src_tensor;
        if (src_seq_len > num_tokens) {
            src_to_copy = uu::make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
        }

        uint32_t block_size = block_manager->get_block_size();
        uint32_t start_pos = current_kv_position;
        uint32_t end_pos = current_kv_position + num_tokens;
        uint32_t start_block_idx = start_pos / block_size;
        uint32_t end_block_idx = (end_pos - 1) / block_size;

        auto allocated_blocks = block_manager->get_allocated_blocks();
        uint32_t blocks_needed = end_block_idx + 1;
        while (allocated_blocks.size() < blocks_needed) {
            auto new_block_id = block_manager->allocate_block();
            OPENVINO_ASSERT(new_block_id.has_value(), "Failed to allocate block for KV cache — pool exhausted");
            allocated_blocks = block_manager->get_allocated_blocks();
        }

        uint32_t tokens_written = 0;
        for (uint32_t block_idx = start_block_idx; block_idx <= end_block_idx; ++block_idx) {
            uint32_t block_id = allocated_blocks[block_idx];
            auto block_tensor = block_manager->get_block_tensor(block_id);

            uint32_t block_start_pos = block_idx * block_size;
            uint32_t write_start_in_block = (start_pos > block_start_pos) ? (start_pos - block_start_pos) : 0;
            uint32_t write_end_in_block = std::min(end_pos - block_start_pos, block_size);
            uint32_t tokens_in_this_block = write_end_in_block - write_start_in_block;

            auto dst_slice = uu::make_tensor_slice(block_tensor, kv_dim, write_start_in_block, write_end_in_block);
            auto src_slice =
                uu::make_tensor_slice(src_to_copy, kv_dim, tokens_written, tokens_written + tokens_in_this_block);
            uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);

            tokens_written += tokens_in_this_block;
            block_manager->update_block_tokens(block_id, write_end_in_block);
        }
        OPENVINO_ASSERT(tokens_written == num_tokens, "Mismatch: wrote ", tokens_written, " but expected ", num_tokens);
    }
}

void BlockKVCacheExtension::copy_prefill_outputs_to_blocks(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const PortsMap& prefill_out_ports,
    uint32_t num_tokens,
    bool v_transposed,
    uint32_t kv_position) {
    LOG_DEBUG("Copying prefill outputs to blocks: num_tokens=" << num_tokens << " kv_position=" << kv_position);
    copy_outputs_to_blocks(prefill_request, prefill_out_ports, num_tokens, v_transposed, kv_position);
}

void BlockKVCacheExtension::commit_generate_kv_and_rebind(
    uint32_t old_num_tokens,
    uint32_t new_num_tokens,
    uint32_t input_tokens_len,
    const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request,
    const PortsMap& kvcache_out_ports,
    const PortsMap& kvcache_in_ports) {
    // Step 1: Write the newly generated token(s) into block storage (partial block, copy-based)
    copy_outputs_to_blocks(kvcache_request,
                           kvcache_out_ports,
                           input_tokens_len,
                           m_compiled_model->m_kvcache_desc.v_tensors_transposed_gen,
                           old_num_tokens);

    // Step 2: Re-bind model inputs if we crossed a block boundary or need tail update
    update_generate_bindings(old_num_tokens, new_num_tokens, kvcache_request, kvcache_in_ports);
}

// ============================================================================
// Private initialization helpers
// ============================================================================

BlockKVCacheExtension::BlockShapeInfo BlockKVCacheExtension::find_block_shapes(const PortsMap& prefill_in_ports) const {
    BlockShapeInfo shapes;
    for (const auto& [name, port] : prefill_in_ports) {
        if (name.find("block_tail") != std::string::npos) {
            continue;
        }
        if (name.find("_block_") != std::string::npos) {
            if (!shapes.found_key && name.find("key_block") != std::string::npos) {
                shapes.key_shape = port.get_shape();
                shapes.elem_type = port.get_element_type();
                shapes.found_key = true;
                LOG_DEBUG("Detected key block shape: " << shapes.key_shape << ", type: " << shapes.elem_type);
            }
            if (!shapes.found_value && name.find("value_block") != std::string::npos) {
                shapes.value_shape = port.get_shape();
                shapes.found_value = true;
                LOG_DEBUG("Detected value block shape: " << shapes.value_shape);
            }
            if (shapes.found_key && shapes.found_value) {
                break;
            }
        }
    }
    return shapes;
}

BlockKVCacheExtension::DummyTensors BlockKVCacheExtension::allocate_dummy_block_tensors(
    const BlockShapeInfo& shapes) const {
    DummyTensors dummies;
    dummies.key_tensor =
        ov::npuw::util::allocMem(shapes.elem_type, shapes.key_shape, "NPU", m_compiled_model->get_plugin());
    LOG_INFO("Allocated shared dummy key tensor: " << shapes.key_shape << " (" << dummies.key_tensor->get_byte_size()
                                                   << " bytes)");

    if (shapes.found_value && shapes.value_shape != shapes.key_shape) {
        dummies.value_tensor =
            ov::npuw::util::allocMem(shapes.elem_type, shapes.value_shape, "NPU", m_compiled_model->get_plugin());
        LOG_INFO("Allocated shared dummy value tensor: " << shapes.value_shape << " ("
                                                         << dummies.value_tensor->get_byte_size() << " bytes)");
    } else {
        dummies.value_tensor = dummies.key_tensor;  // Reuse key tensor
    }
    return dummies;
}

void BlockKVCacheExtension::set_dummy_tensors_to_all_requests(
    const DummyTensors& dummies,
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
    const PortsMap& prefill_in_ports,
    const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports) {
    size_t prefill_block_count = set_dummy_block_tensors(
        prefill_in_ports,
        [&prefill_request](const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
            prefill_request->set_tensor(port, tensor);
        },
        dummies.key_tensor,
        dummies.value_tensor);
    LOG_INFO("Set " << prefill_block_count << " prefill numbered block inputs to shared dummy tensors");

    size_t generate_block_count = 0;
    for (auto& generate_request : generate_requests) {
        generate_block_count += set_dummy_block_tensors(
            gen_variant_in_ports.at(generate_request),
            [&generate_request](const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
                generate_request->set_tensor(port, tensor);
            },
            dummies.key_tensor,
            dummies.value_tensor);
    }
    LOG_INFO("Set " << generate_block_count << " generate numbered block inputs to shared dummy tensors");
}

BlockKVCacheExtension::LayerBlockNames BlockKVCacheExtension::parse_block_inputs_structure(
    const PortsMap& prefill_in_ports) const {
    LayerBlockNames layer_blocks;

    for (const auto& [name, port] : prefill_in_ports) {
        if (name.find("_block_") == std::string::npos) {
            continue;
        }
        std::regex layer_regex(R"(past_key_values\.(\d+)\.)");
        std::smatch match;
        uint32_t layer_idx = 0;
        if (std::regex_search(name, match, layer_regex) && match.size() > 1) {
            layer_idx = static_cast<uint32_t>(std::stoi(match[1].str()));
        } else {
            LOG_DEBUG("WARNING: Could not parse layer index from " << name << ", using default 0");
        }

        bool is_key = (name.find("key_block") != std::string::npos);
        if (is_key) {
            layer_blocks[layer_idx].first.push_back(name);
        } else {
            layer_blocks[layer_idx].second.push_back(name);
        }
    }
    return layer_blocks;
}

void BlockKVCacheExtension::create_block_managers_and_helpers(
    const LayerBlockNames& layer_blocks,
    const PortsMap& prefill_in_ports,
    const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
    const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports) {
    const uint32_t block_size = static_cast<uint32_t>(m_compiled_model->m_prefill_chunk_size);
    const uint32_t max_blocks = (m_compiled_model->m_kvcache_desc.total_size + block_size - 1) / block_size;

    LOG_INFO("Block configuration: size=" << block_size << " tokens, max_blocks=" << max_blocks);

    for (const auto& [layer_idx, block_names_pair] : layer_blocks) {
        const auto& key_block_names = block_names_pair.first;
        const auto& value_block_names = block_names_pair.second;

        if (key_block_names.empty() && value_block_names.empty()) {
            continue;
        }

        KVCacheBlockManager::LayerBlockManagers layer_managers;
        // Use block_0 name specifically to deterministically get the full-block shape.
        const std::string layer_idx_str_local = std::to_string(layer_idx);

        if (!key_block_names.empty()) {
            // Deterministically pick block_0 (full-block shape, not a smaller tail block)
            const std::string key_block0_name = make_numbered_block_input_name("key", layer_idx_str_local, 0);
            auto first_key_port = prefill_in_ports.at(key_block0_name);
            layer_managers.key_manager = std::make_unique<KVCacheBlockManager>(block_size,
                                                                               max_blocks,
                                                                               first_key_port.get_shape(),
                                                                               first_key_port.get_element_type(),
                                                                               "NPU",
                                                                               m_compiled_model->get_plugin());
        }

        if (!value_block_names.empty()) {
            // Deterministically pick block_0 (full-block shape, not a smaller tail block)
            const std::string value_block0_name = make_numbered_block_input_name("value", layer_idx_str_local, 0);
            auto first_value_port = prefill_in_ports.at(value_block0_name);
            layer_managers.value_manager = std::make_unique<KVCacheBlockManager>(block_size,
                                                                                 max_blocks,
                                                                                 first_value_port.get_shape(),
                                                                                 first_value_port.get_element_type(),
                                                                                 "NPU",
                                                                                 m_compiled_model->get_plugin());
        }

        m_kv_cache_block_managers[layer_idx] = std::move(layer_managers);

        // Pre-compute BlockBindingHelpers for this layer × all generate variants
        const std::string layer_idx_str = std::to_string(layer_idx);
        const auto& layer_managers_ref = m_kv_cache_block_managers.at(layer_idx);

        for (const auto& generate_request : generate_requests) {
            const auto& variant_in_ports = gen_variant_in_ports.at(generate_request);
            auto& variant_layer_helpers = m_variant_block_binding_helpers[generate_request];

            LayerBlockBindingHelpers layer_helpers;

            if (layer_managers_ref.key_manager) {
                uint32_t max_blocks_key = layer_managers_ref.key_manager->get_max_blocks();
                auto key_ports = partition_layer_block_ports("key", layer_idx_str, variant_in_ports, max_blocks_key);
                layer_helpers.key_helper =
                    BlockBindingHelper::from_ports(std::move(key_ports.numbered),
                                                   std::move(key_ports.tail),
                                                   layer_managers_ref.key_manager->get_block_size());
            }

            if (layer_managers_ref.value_manager) {
                uint32_t max_blocks_value = layer_managers_ref.value_manager->get_max_blocks();
                auto value_ports =
                    partition_layer_block_ports("value", layer_idx_str, variant_in_ports, max_blocks_value);
                layer_helpers.value_helper =
                    BlockBindingHelper::from_ports(std::move(value_ports.numbered),
                                                   std::move(value_ports.tail),
                                                   layer_managers_ref.value_manager->get_block_size());
            }

            variant_layer_helpers[layer_idx] = std::move(layer_helpers);
        }
    }
}

}  // namespace npuw
}  // namespace ov
