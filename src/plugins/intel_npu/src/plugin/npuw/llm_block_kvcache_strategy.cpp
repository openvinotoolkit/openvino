// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_block_kvcache_strategy.hpp"

#include <cmath>
#include <regex>

#include "infer_request_utils.hpp"
#include "llm_compiled_model.hpp"
#include "llm_infer_base_request.hpp"
#include "llm_infer_request.hpp"
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

// Classify a port name as a numbered block KV param (non-contiguous, non-tail).
// Returns: true  => numbered key block (key_block_N)
//          false => numbered value block (value_block_N)
//          nullopt => skip (not a KV param, contiguous, or tail)
std::optional<bool> classify_numbered_block_param(const std::string& name) {
    namespace uu = ov::npuw::util;
    const bool is_key = uu::isPastKeyParam(name);
    const bool is_value = uu::isPastValueParam(name);
    if (!is_key && !is_value) {
        return std::nullopt;
    }
    if (uu::isPastKeyValuesKeyContiguous(name).has_value() || uu::isPastKeyValuesValueContiguous(name).has_value()) {
        return std::nullopt;
    }
    if (name.find("block_tail") != std::string::npos) {
        return std::nullopt;
    }
    return is_key;
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
        const auto kv_type = classify_numbered_block_param(name);
        if (!kv_type.has_value()) {
            continue;
        }
        set_tensor_fn(port, kv_type.value() ? dummy_key_tensor : dummy_value_tensor);
        block_count++;
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

// Shapes and element type of one key/value block pair — derived from the first
// numbered block port found during initialization.
struct BlockShapeInfo {
    ov::Shape key_shape;
    ov::Shape value_shape;
    ov::element::Type elem_type;
    bool found_key = false;
    bool found_value = false;
};

// Scan prefill input ports to discover the shape and element type of block tensors.
BlockShapeInfo find_block_shapes(const ov::npuw::LLMBlockKVCacheStrategy::PortsMap& prefill_in_ports) {
    BlockShapeInfo shapes;
    for (const auto& [name, port] : prefill_in_ports) {
        const auto kv_type = classify_numbered_block_param(name);
        if (!kv_type.has_value()) {
            continue;
        }
        const bool is_key = kv_type.value();
        if (!shapes.found_key && is_key) {
            shapes.key_shape = port.get_shape();
            shapes.elem_type = port.get_element_type();
            shapes.found_key = true;
            LOG_DEBUG("Detected key block shape: " << shapes.key_shape << ", type: " << shapes.elem_type);
        }
        if (!shapes.found_value && !is_key) {
            shapes.value_shape = port.get_shape();
            shapes.found_value = true;
            LOG_DEBUG("Detected value block shape: " << shapes.value_shape);
        }
        if (shapes.found_key && shapes.found_value) {
            break;
        }
    }
    return shapes;
}

// Set dummy tensors on all numbered block input ports of a single inference request.
size_t set_dummy_tensors_to_request(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                    const ov::npuw::LLMBlockKVCacheStrategy::PortsMap& ports_map,
                                    const ov::npuw::LLMBlockKVCacheStrategy::DummyTensors& dummies) {
    return set_dummy_block_tensors(
        ports_map,
        [&request](const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {
            request->set_tensor(port, tensor);
        },
        dummies.key_tensor,
        dummies.value_tensor);
}

}  // anonymous namespace

namespace ov {
namespace npuw {

// ============================================================================
// LLMKVCacheStrategy interface
// ============================================================================

void LLMBlockKVCacheStrategy::on_initialize() {
    LOG_INFO("=== Initializing Block-based KV Cache Managers ===");

    const auto& prefill_in_ports = m_req.m_prefill_in_ports;
    const auto& prefill_out_ports = m_req.m_prefill_out_ports;

    auto block_shapes = find_block_shapes(prefill_in_ports);
    if (block_shapes.found_key) {
        // Dummy tensor optimization: share one dummy tensor per shape across all block inputs.
        // Store in m_dummy_tensors so on_reset() can restore ports to release block tensor refs.
        const auto& plugin = m_req.m_npuw_llm_compiled_model->get_plugin();
        m_dummy_tensors.key_tensor =
            ov::npuw::util::allocMem(block_shapes.elem_type, block_shapes.key_shape, m_req.m_pre_alloc_device, plugin);
        LOG_INFO("Allocated shared dummy key tensor: " << block_shapes.key_shape << " ("
                                                       << m_dummy_tensors.key_tensor->get_byte_size() << " bytes)");
        if (block_shapes.found_value && block_shapes.value_shape != block_shapes.key_shape) {
            m_dummy_tensors.value_tensor = ov::npuw::util::allocMem(block_shapes.elem_type,
                                                                    block_shapes.value_shape,
                                                                    m_req.m_pre_alloc_device,
                                                                    plugin);
            LOG_INFO("Allocated shared dummy value tensor: "
                     << block_shapes.value_shape << " (" << m_dummy_tensors.value_tensor->get_byte_size() << " bytes)");
        } else {
            m_dummy_tensors.value_tensor = m_dummy_tensors.key_tensor;
        }
        set_dummy_tensors_to_all_requests();
    }

    // Create block managers and pre-compute per-variant binding helpers
    create_block_managers_and_helpers(prefill_in_ports, m_req.m_generate_requests, m_req.m_generate_variant_in_ports);

    // Snapshot original prefill output tensors (for restore_prefill_output_buffers()) and
    // build m_output_kv_info (output_name → layer/kv) to avoid per-call regex.
    for (const auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
        const std::string layer_str = std::to_string(layer_idx);
        for (const bool is_key : {true, false}) {
            const std::string kv_type = is_key ? "key" : "value";
            const std::string output_name = "present." + layer_str + "." + kv_type;
            m_output_kv_info[output_name] = {layer_idx, is_key};
            auto port_it = prefill_out_ports.find(output_name);
            if (port_it != prefill_out_ports.end()) {
                m_prefill_original_output_tensors[output_name] = m_req.m_prefill_request->get_tensor(port_it->second);
            }
        }
    }
    LOG_INFO("Snapshotted " << m_prefill_original_output_tensors.size() << " original prefill output tensors");
    LOG_INFO("=== Block-based KV Cache Initialization Complete ===");
}

void LLMBlockKVCacheStrategy::set_dummy_tensors_to_all_requests() {
    size_t prefill_count =
        set_dummy_tensors_to_request(m_req.m_prefill_request, m_req.m_prefill_in_ports, m_dummy_tensors);
    LOG_INFO("Set " << prefill_count << " prefill numbered block inputs to shared dummy tensors");

    size_t gen_count = 0;
    for (const auto& generate_request : m_req.m_generate_requests) {
        gen_count += set_dummy_tensors_to_request(generate_request,
                                                  m_req.m_generate_variant_in_ports.at(generate_request),
                                                  m_dummy_tensors);
    }
    LOG_INFO("Set " << gen_count << " generate numbered block inputs to shared dummy tensors");
}

void LLMBlockKVCacheStrategy::on_reset(uint32_t next_prompt_length) {
    // When no block inputs were found during on_initialize() there are no block
    // tensors to release and no dummy tensors to propagate — nothing to do.
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    // ── Step 1: drop prefill OUTPUT port refs ────────────────────────────────────────
    // Only needed when the last prefill chunk used the zero-copy path, which redirects
    // prefill output ports directly to block tensors.  The copy path leaves output ports
    // pointing to the original buffers, so no restore is needed in that case.
    {
        if (m_zero_copy_last_chunk) {
            restore_prefill_output_buffers(m_req.m_prefill_request, m_req.m_prefill_out_ports);
            m_zero_copy_last_chunk = false;
        }
    }

    // ── Step 2: drop prefill/generate INPUT port refs ────────────────────────────────
    // Only the prefill request and the currently-selected generate variant were ever
    // re-bound to live block tensors.  Other variants remain on dummy tensors from
    // on_initialize() and need no action.
    {
        set_dummy_tensors_to_request(m_req.m_prefill_request, m_req.m_prefill_in_ports, m_dummy_tensors);
        set_dummy_tensors_to_request(m_req.m_kvcache_request,
                                     m_req.m_generate_variant_in_ports.at(m_req.m_kvcache_request),
                                     m_dummy_tensors);
    }

    // ── Step 3: propagate dummies into sub-requests ──────────────────────────────────
    // Sub-requests hold their own SoPtr to block tensors and only refresh them at the
    // next infer() call.  Push the dummies set above into the current variant's
    // sub-requests so device memory is freed immediately.  Other variants were never
    // bound to live block tensors, so they do not need propagation.
    {
        auto it =
            std::find(m_req.m_generate_requests.begin(), m_req.m_generate_requests.end(), m_req.m_kvcache_request);
        if (it != m_req.m_generate_requests.end()) {
            const size_t idx = static_cast<size_t>(std::distance(m_req.m_generate_requests.begin(), it));
            m_req.m_generate_base_requests[idx]->propagate_params_to_subrequests();
        }
    }

    // ── Step 4: release block tensors ────────────────────────────────────────────────
    // All SoPtr references to block tensors have been dropped above; calling release()
    // here will actually return device memory to the allocator.
    // Keep ceil(next_prompt_length / block_size) blocks warm to avoid re-allocating
    // them on the next prefill.  Pass 0 when the next prompt length is unknown.
    {
        const uint32_t keep_warm_blocks =
            (m_block_size > 0 && next_prompt_length > 0)
                ? static_cast<uint32_t>(
                      std::ceil(static_cast<double>(next_prompt_length) / static_cast<double>(m_block_size)))
                : 0u;
        for (auto& [layer_idx, layer_managers] : m_kv_cache_block_managers) {
            if (layer_managers.key_manager) {
                layer_managers.key_manager->release(keep_warm_blocks);
            }
            if (layer_managers.value_manager) {
                layer_managers.value_manager->release(keep_warm_blocks);
            }
        }
    }
}

void LLMBlockKVCacheStrategy::on_prefill_chunk_begin(uint32_t current_prompts_len) {
    OPENVINO_ASSERT(!m_kv_cache_block_managers.empty(),
                    "Block-based KV cache is enabled but no block inputs were found. "
                    "Ensure NPUW_LLM_ENABLE_BLOCK_BASED_KV_CACHE is only set together with "
                    "chunk prefill and a supported attention pattern (HFA or Pyramid).");

    load_past_kv_blocks_to_prefill(m_req.m_prefill_request, m_req.m_prefill_in_ports);

    if (current_prompts_len % m_block_size == 0) {
        m_zero_copy_last_chunk = redirect_prefill_outputs_to_new_blocks(m_req.m_prefill_request,
                                                                        m_req.m_prefill_out_ports,
                                                                        current_prompts_len);
    } else {
        m_zero_copy_last_chunk = false;
    }
    if (!m_zero_copy_last_chunk) {
        LOG_DEBUG("Chunk not block-aligned (" << current_prompts_len << " % " << m_block_size
                                              << " != 0), falling back to copy path");
        // Restore original output buffers so the model does not write into blocks that were
        // redirected by a previous zero-copy chunk.
        restore_prefill_output_buffers(m_req.m_prefill_request, m_req.m_prefill_out_ports);
    }
}

void LLMBlockKVCacheStrategy::on_prefill_chunk_done(uint32_t current_prompts_len, bool /*is_last*/) {
    if (m_zero_copy_last_chunk) {
        // Zero-copy: model wrote directly into blocks; redirect_prefill_outputs_to_new_blocks()
        // already updated metadata — nothing to do.
        LOG_DEBUG("Zero-copy prefill complete - no post-inference work needed");
    } else {
        // Copy path: copy from the model's output buffer into blocks.
        // write start = num_stored_tokens (already incremented) - current_prompts_len.
        const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
        const uint32_t write_start = kvcache_desc.num_stored_tokens - current_prompts_len;
        LOG_DEBUG("Copying prefill outputs to blocks: num_tokens=" << current_prompts_len
                                                                   << " kv_position=" << write_start);
        const bool v_transposed = kvcache_desc.v_tensors_transposed_pre;
        copy_outputs_to_blocks(m_req.m_prefill_request,
                               m_req.m_prefill_out_ports,
                               current_prompts_len,
                               v_transposed,
                               write_start);
    }
}

void LLMBlockKVCacheStrategy::on_generate_kv_init() {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }

    LOG_DEBUG("=== Initial Block Binding for Generate Model ===");
    LOG_BLOCK();

    // Initial block binding (called once per conversation on the first generate step):
    //  Numbered blocks (block_0…block_N): zero-copy via set_tensor(); subsequent generate steps
    //    write directly into the shared BlockManager buffer — no re-binding needed.
    //  Tail block (block_tail port): copy-based; refreshed by update_generate_bindings() each step.

    auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const auto& variant_layer_helpers = m_variant_block_binding_helpers.at(m_req.m_kvcache_request);

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
            const uint32_t block_id = allocated_blocks[block_idx];
            const auto block_tensor = manager->get_block_tensor(block_id);
            if (helper.should_treat_as_tail(static_cast<uint32_t>(block_idx))) {
                copy_block_to_tail_input(block_tensor,
                                         0u,
                                         manager->get_block_tokens(block_id),
                                         kv_dim,
                                         helper.tail_input_port.value(),
                                         m_req.m_kvcache_request);
                LOG_VERB("Bound tail " << kv_type_name << " layer_" << layer_idx << " block_" << block_idx);
            } else if (block_idx < helper.numbered_input_ports.size()) {
                m_req.m_kvcache_request->set_tensor(helper.numbered_input_ports[block_idx], block_tensor);
                LOG_VERB("Bound numbered " << kv_type_name << " layer_" << layer_idx << " block_" << block_idx);
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

    LOG_DEBUG("Initial binding complete.");
}

void LLMBlockKVCacheStrategy::on_generate_step_done(uint32_t input_tokens_len) {
    const auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const uint32_t tokens_after = kvcache_desc.num_stored_tokens;
    const uint32_t tokens_before = tokens_after - input_tokens_len;
    copy_outputs_to_blocks(m_req.m_kvcache_request,
                           m_req.m_kvcache_out_ports,
                           input_tokens_len,
                           kvcache_desc.v_tensors_transposed_gen,
                           tokens_before);
    update_generate_bindings(tokens_before, tokens_after, m_req.m_kvcache_request);
}

// ============================================================================
// Private: prefill path primitives
// ============================================================================

void LLMBlockKVCacheStrategy::load_past_kv_blocks_to_prefill(
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

bool LLMBlockKVCacheStrategy::redirect_prefill_outputs_to_new_blocks(
    const std::shared_ptr<ov::IAsyncInferRequest>& prefill_request,
    const PortsMap& prefill_out_ports,
    uint32_t num_new_tokens) {
    if (m_kv_cache_block_managers.empty()) {
        return false;
    }
    auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    uint32_t current_position = kvcache_desc.num_stored_tokens;
    if (current_position % m_block_size != 0) {
        return false;
    }

    // Zero-copy: bind prefill output ports directly to fresh blocks BEFORE inference so the
    // model writes KV cache straight into BlockManager memory with no post-inference copy.
    // Requires current_position to be block-aligned and num_new_tokens to fit in one block;
    // callers must call restore_prefill_output_buffers() and fall back to copy_outputs_to_blocks()
    // when this does not hold.
    LOG_DEBUG("=== Redirecting Prefill Outputs to New Blocks (Zero-Copy) ===");
    LOG_DEBUG("Pre-allocating blocks for " << num_new_tokens << " new tokens");

    size_t total_blocks_allocated = 0;
    size_t total_outputs_bound = 0;

    auto process_kv_blocks = [&](uint32_t layer_idx, KVCacheBlockManager* manager, const char* kv_type_name) -> bool {
        uint32_t start_pos = current_position;
        uint32_t end_pos = current_position + num_new_tokens;
        uint32_t start_block_idx = start_pos / m_block_size;
        uint32_t end_block_idx = (end_pos - 1) / m_block_size;

        // Alignment postconditions for zero-copy
        OPENVINO_ASSERT(start_pos % m_block_size == 0,
                        "Zero-copy prefill requires block-aligned position. ",
                        "current_position=",
                        start_pos,
                        ", block_size=",
                        m_block_size);
        OPENVINO_ASSERT(start_block_idx == end_block_idx,
                        "Zero-copy prefill requires writing to a single block. ",
                        "start_block=",
                        start_block_idx,
                        ", end_block=",
                        end_block_idx,
                        ", num_new_tokens=",
                        num_new_tokens,
                        ", block_size=",
                        m_block_size);

        uint32_t tokens_in_block = end_pos - (start_block_idx * m_block_size);
        OPENVINO_ASSERT(tokens_in_block == m_block_size || tokens_in_block == num_new_tokens,
                        "Zero-copy prefill requires writing full blocks. ",
                        "tokens_in_block=",
                        tokens_in_block,
                        ", block_size=",
                        m_block_size,
                        ", num_new_tokens=",
                        num_new_tokens);

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
            uint32_t block_start_pos = start_block_idx * m_block_size;
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
    return true;
}

void LLMBlockKVCacheStrategy::restore_prefill_output_buffers(
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
    // Fix: restore the original model output buffers (snapshotted in on_initialize()).
    // The model then writes into its own buffers; copy_outputs_to_blocks() copies
    // the result into the correct blocks afterwards.
    OPENVINO_ASSERT(!m_prefill_original_output_tensors.empty(),
                    "Original output tensors were not snapshotted during on_initialize().");
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
}

// ============================================================================
// Private: generate path — update bindings after each step
// ============================================================================

void LLMBlockKVCacheStrategy::update_generate_bindings(uint32_t old_num_tokens,
                                                       uint32_t new_num_tokens,
                                                       const std::shared_ptr<ov::IAsyncInferRequest>& kvcache_request) {
    if (m_kv_cache_block_managers.empty()) {
        return;
    }
    LOG_DEBUG("=== Update Block Bindings After Generate ===");
    LOG_BLOCK();

    auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    const auto& variant_layer_helpers = m_variant_block_binding_helpers.at(kvcache_request);

    auto update_blocks = [&](uint32_t layer_idx,
                             KVCacheBlockManager* manager,
                             const BlockBindingHelper& helper,
                             uint32_t kv_dim,
                             const char* kv_type_name) {
        const auto allocated_blocks = manager->get_allocated_blocks();
        if (allocated_blocks.empty()) {
            return;
        }
        NPUW_ASSERT(old_num_tokens > 0);
        // update_generate_bindings is only called after at least one prefill has run,
        // so old_num_tokens must be positive.
        const uint32_t old_block_idx = helper.get_block_index_for_position(old_num_tokens - 1);
        const uint32_t final_block_idx = helper.get_block_index_for_position(new_num_tokens - 1);

        if (final_block_idx > old_block_idx) {
            // A block boundary was crossed: bind each newly entered block.
            // In standard decoding (input_tokens_len=1) this loop runs exactly once.
            // In speculative decoding it may run more times if input_tokens_len > block_size.
            //   - Numbered blocks: zero-copy via set_tensor().
            //   - Tail block:      full copy from token 0 of the block.
            for (uint32_t bidx = old_block_idx + 1;
                 bidx <= final_block_idx && bidx < static_cast<uint32_t>(allocated_blocks.size());
                 ++bidx) {
                const uint32_t block_id = allocated_blocks[bidx];
                if (helper.should_treat_as_tail(bidx)) {
                    copy_block_to_tail_input(manager->get_block_tensor(block_id),
                                             0u,
                                             manager->get_block_tokens(block_id),
                                             kv_dim,
                                             helper.tail_input_port.value(),
                                             kvcache_request);
                    LOG_VERB("Updated tail " << kv_type_name << " layer_" << layer_idx << " block_" << bidx
                                             << " (crossed block boundary)");
                } else if (bidx < helper.numbered_input_ports.size()) {
                    kvcache_request->set_tensor(helper.numbered_input_ports[bidx], manager->get_block_tensor(block_id));
                    LOG_VERB("Bound new numbered " << kv_type_name << " layer_" << layer_idx << " block_" << bidx);
                }
            }
        } else {
            // No block boundary crossed: old and new tokens both reside in the same block.
            // Numbered blocks share memory with BlockManager — no action needed.
            // Tail block requires an incremental copy for the newly added tokens only.
            if (helper.should_treat_as_tail(final_block_idx)) {
                const uint32_t block_id = allocated_blocks[final_block_idx];
                const uint32_t block_start = final_block_idx * helper.block_size;
                const uint32_t prev_tokens = old_num_tokens - block_start;
                const uint32_t new_tokens = manager->get_block_tokens(block_id);
                copy_block_to_tail_input(manager->get_block_tensor(block_id),
                                         prev_tokens,
                                         new_tokens,
                                         kv_dim,
                                         helper.tail_input_port.value(),
                                         kvcache_request);
                LOG_VERB("Refreshed tail " << kv_type_name << " layer_" << layer_idx << " block_" << final_block_idx
                                           << " [" << prev_tokens << ".." << new_tokens << "]");
            }
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
}

// ============================================================================
// Private: KV copy engine
// ============================================================================

void LLMBlockKVCacheStrategy::copy_outputs_to_blocks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                                     const PortsMap& src_ports,
                                                     uint32_t num_tokens,
                                                     bool v_transposed,
                                                     uint32_t current_kv_position) {
    namespace uu = ov::npuw::util;
    auto& kvcache_desc = m_req.m_npuw_llm_compiled_model->m_kvcache_desc;
    auto& compiled = request->get_compiled_model();

    for (std::size_t i = LLMInferBaseRequest::layer_ids::kStartOutputKVCacheLayers; i < compiled->outputs().size();
         ++i) {
        const auto& output_name = compiled->outputs()[i].get_any_name();
        // Classify output port via pre-computed map (avoids per-call regex).
        auto info_it = m_output_kv_info.find(output_name);
        if (info_it == m_output_kv_info.end()) {
            continue;
        }
        const bool is_key = info_it->second.is_key;
        const uint32_t layer_idx = info_it->second.layer_idx;

        auto it = m_kv_cache_block_managers.find(layer_idx);
        if (it == m_kv_cache_block_managers.end()) {
            continue;
        }

        auto& layer_managers = it->second;
        auto& block_manager = is_key ? layer_managers.key_manager : layer_managers.value_manager;
        const uint32_t kv_dim = (!is_key && v_transposed) ? 3u : kvcache_desc.dim;

        auto src_tensor = request->get_tensor(src_ports.at(output_name));
        uint32_t src_seq_len = static_cast<uint32_t>(src_tensor->get_shape()[kv_dim]);
        OPENVINO_ASSERT(num_tokens <= src_seq_len, "num_tokens (", num_tokens, ") > src_seq_len (", src_seq_len, ")");

        auto src_to_copy = src_tensor;
        if (src_seq_len > num_tokens) {
            src_to_copy = uu::make_tensor_slice(src_tensor, kv_dim, src_seq_len - num_tokens, src_seq_len);
        }

        const uint32_t start_pos = current_kv_position;
        const uint32_t end_pos = current_kv_position + num_tokens;
        const uint32_t start_block_idx = start_pos / m_block_size;
        const uint32_t end_block_idx = (end_pos - 1) / m_block_size;

        // Allocate any blocks that do not yet exist, then fetch the final list once.
        auto allocated_blocks = block_manager->get_allocated_blocks();
        for (uint32_t b = static_cast<uint32_t>(allocated_blocks.size()); b <= end_block_idx; ++b) {
            OPENVINO_ASSERT(block_manager->allocate_block().has_value(),
                            "Failed to allocate block for KV cache — pool exhausted");
        }
        allocated_blocks = block_manager->get_allocated_blocks();

        // Write tokens across blocks.
        // The first block may be partially filled: start writing at (start_pos % block_size).
        // Every subsequent block is written from position 0 up to min(remaining, block_size).
        uint32_t tokens_written = 0;
        for (uint32_t block_idx = start_block_idx; block_idx <= end_block_idx; ++block_idx) {
            const uint32_t block_id = allocated_blocks[block_idx];
            const auto block_tensor = block_manager->get_block_tensor(block_id);
            const uint32_t write_start = (block_idx == start_block_idx) ? (start_pos % m_block_size) : 0u;
            const uint32_t write_end = std::min(end_pos - block_idx * m_block_size, m_block_size);
            const uint32_t count = write_end - write_start;
            const auto dst_slice = uu::make_tensor_slice(block_tensor, kv_dim, write_start, write_end);
            const auto src_slice = uu::make_tensor_slice(src_to_copy, kv_dim, tokens_written, tokens_written + count);
            uu::copy_tensor_by_dim(src_slice, dst_slice, kv_dim, kv_dim);
            tokens_written += count;
            block_manager->update_block_tokens(block_id, write_end);
        }
        OPENVINO_ASSERT(tokens_written == num_tokens, "Mismatch: wrote ", tokens_written, " but expected ", num_tokens);
    }
}

// ============================================================================
// Private: initialization helpers
// ============================================================================

void LLMBlockKVCacheStrategy::create_block_managers_and_helpers(
    const PortsMap& prefill_in_ports,
    const std::vector<std::shared_ptr<ov::IAsyncInferRequest>>& generate_requests,
    const std::unordered_map<std::shared_ptr<ov::IAsyncInferRequest>, PortsMap>& gen_variant_in_ports) {
    const auto& compiled_model = m_req.m_npuw_llm_compiled_model;
    const uint32_t block_size = static_cast<uint32_t>(compiled_model->m_prefill_chunk_size);
    m_block_size = block_size;
    const uint32_t max_blocks = (compiled_model->m_kvcache_desc.total_size + block_size - 1) / block_size;

    LOG_INFO("Block configuration: size=" << block_size << " tokens, max_blocks=" << max_blocks);

    // -------------------------------------------------------------------------
    // Phase 1: Single scan — discover which layers have key/value block ports.
    // Records both numbered blocks and tail ports so Phase 2 can cross-validate.
    // -------------------------------------------------------------------------
    struct LayerKVPresence {
        bool has_key_numbered_block = false;    // at least one key_block_N found
        bool has_value_numbered_block = false;  // at least one value_block_N found
        bool has_key_tail_block = false;        // key_block_tail found
        bool has_value_tail_block = false;      // value_block_tail found
    };
    std::map<uint32_t, LayerKVPresence> layer_presence;

    static const std::regex layer_regex(R"(past_key_values\.(\d+)\.)");
    for (const auto& [name, port] : prefill_in_ports) {
        namespace uu = ov::npuw::util;
        if (!uu::isPastKeyParam(name) && !uu::isPastValueParam(name)) {
            continue;
        }
        std::smatch match;
        if (!std::regex_search(name, match, layer_regex) || match.size() <= 1) {
            OPENVINO_THROW("NPUW block KV cache: could not parse layer index from port name: ", name);
        }
        const uint32_t layer_idx = static_cast<uint32_t>(std::stoi(match[1].str()));
        auto& presence = layer_presence[layer_idx];

        const bool is_tail = name.find("block_tail") != std::string::npos;
        const bool is_key = uu::isPastKeyParam(name);
        if (is_tail) {
            if (is_key)
                presence.has_key_tail_block = true;
            else
                presence.has_value_tail_block = true;
        } else if (classify_numbered_block_param(name).has_value()) {
            if (is_key)
                presence.has_key_numbered_block = true;
            else
                presence.has_value_numbered_block = true;
        }
        // else: contiguous param — skip
    }

    // -------------------------------------------------------------------------
    // Phase 2: For each layer, validate invariants, create block managers,
    //          and pre-compute per-generate-variant binding helpers.
    //
    // Invariants (broken SplitKVCacheIntoBlocks would violate these):
    //   A. tail-only layer: has_key_tail without has_key (or same for value)
    //   B. missing block_0: has_key/has_value but key/value_block_0 is absent
    // -------------------------------------------------------------------------
    for (const auto& [layer_idx, presence] : layer_presence) {
        const std::string layer_idx_str = std::to_string(layer_idx);

        // Invariant A: tail must not exist without numbered blocks
        OPENVINO_ASSERT(!presence.has_key_tail_block || presence.has_key_numbered_block,
                        "NPUW block KV cache: layer ",
                        layer_idx,
                        " has key_block_tail but no key numbered blocks. "
                        "SplitKVCacheIntoBlocks transformation may be broken.");
        OPENVINO_ASSERT(!presence.has_value_tail_block || presence.has_value_numbered_block,
                        "NPUW block KV cache: layer ",
                        layer_idx,
                        " has value_block_tail but no value numbered blocks. "
                        "SplitKVCacheIntoBlocks transformation may be broken.");

        LayerBlockManagers layer_managers;

        if (presence.has_key_numbered_block) {
            // Invariant B: numbered blocks must start at block_0
            const std::string key_block0_name = make_numbered_block_input_name("key", layer_idx_str, 0);
            OPENVINO_ASSERT(prefill_in_ports.count(key_block0_name),
                            "NPUW block KV cache: layer ",
                            layer_idx,
                            " has key blocks but no key_block_0. "
                            "SplitKVCacheIntoBlocks transformation may be broken.");
            auto first_key_port = prefill_in_ports.at(key_block0_name);
            layer_managers.key_manager = std::make_unique<KVCacheBlockManager>(block_size,
                                                                               max_blocks,
                                                                               first_key_port.get_shape(),
                                                                               first_key_port.get_element_type(),
                                                                               m_req.m_pre_alloc_device,
                                                                               compiled_model->get_plugin());
        }
        if (presence.has_value_numbered_block) {
            const std::string value_block0_name = make_numbered_block_input_name("value", layer_idx_str, 0);
            OPENVINO_ASSERT(prefill_in_ports.count(value_block0_name),
                            "NPUW block KV cache: layer ",
                            layer_idx,
                            " has value blocks but no value_block_0. "
                            "SplitKVCacheIntoBlocks transformation may be broken.");
            auto first_value_port = prefill_in_ports.at(value_block0_name);
            layer_managers.value_manager = std::make_unique<KVCacheBlockManager>(block_size,
                                                                                 max_blocks,
                                                                                 first_value_port.get_shape(),
                                                                                 first_value_port.get_element_type(),
                                                                                 m_req.m_pre_alloc_device,
                                                                                 compiled_model->get_plugin());
        }

        m_kv_cache_block_managers[layer_idx] = std::move(layer_managers);
        const auto& layer_managers_ref = m_kv_cache_block_managers.at(layer_idx);

        for (const auto& generate_request : generate_requests) {
            const auto& variant_in_ports = gen_variant_in_ports.at(generate_request);
            auto& variant_layer_helpers = m_variant_block_binding_helpers[generate_request];
            LayerBlockBindingHelpers layer_helpers;

            if (layer_managers_ref.key_manager) {
                uint32_t max_blocks_key = layer_managers_ref.key_manager->get_max_blocks();
                auto key_ports = partition_layer_block_ports("key", layer_idx_str, variant_in_ports, max_blocks_key);
                layer_helpers.key_helper = BlockBindingHelper::from_ports(std::move(key_ports.numbered),
                                                                          std::move(key_ports.tail),
                                                                          block_size);
            }
            if (layer_managers_ref.value_manager) {
                uint32_t max_blocks_value = layer_managers_ref.value_manager->get_max_blocks();
                auto value_ports =
                    partition_layer_block_ports("value", layer_idx_str, variant_in_ports, max_blocks_value);
                layer_helpers.value_helper = BlockBindingHelper::from_ports(std::move(value_ports.numbered),
                                                                            std::move(value_ports.tail),
                                                                            block_size);
            }
            variant_layer_helpers[layer_idx] = std::move(layer_helpers);
        }
    }
}

}  // namespace npuw
}  // namespace ov
