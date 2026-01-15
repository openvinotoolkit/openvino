// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_executor.hpp"

#include "../compiled_model.hpp"  // For CompiledModel::CompiledModelDesc
#include "../logging.hpp"
#include "../moe_infer_utils.hpp"
#include "moe_types.hpp"  // For MoEIO definition
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"  // For ov::parallel_for

namespace ov {
namespace npuw {
namespace moe {

MoEExecutor::MoEExecutor(ISubrequestAccessor& accessor, ProfilerFn profiler, AllocatorFn allocator)
    : m_accessor(accessor),
      m_profiler(std::move(profiler)),
      m_allocator(std::move(allocator)) {
    LOG_DEBUG("MoEExecutor created");
}

void MoEExecutor::prepare(size_t idx, size_t real_idx, size_t num_sublayers, size_t pool_size) {
    LOG_INFO("Preparing MoE resources for sublayer[" << idx << "] (real_idx=" << real_idx << ")...");
    LOG_BLOCK();

    // Get function body descriptor directly (no need to resolve, caller already did it)
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    if (!desc.moe_experts.has_value()) {
        OPENVINO_THROW("MoEExecutor::prepare called on non-MoE submodel[real_idx=", real_idx, "]");
    }

    const auto& moe_experts = desc.moe_experts.value();

    // Step 1: Initialize MoE config (only once, on first prepare call)
    if (!m_config.is_valid()) {
        LOG_DEBUG("Creating shared MoE config");

        m_config.num_experts = moe_experts.num_experts;
        m_config.num_active_experts = moe_experts.num_active_experts;
        m_config.input_token_count = moe_experts.input_token_count;
        m_config.expert_hidden_dim = moe_experts.expert_hidden_dim;
        m_config.param_mapping = moe_experts._param_mapping;
        m_config.router_scores_idx = moe_experts._router_scores_idx;
        m_config.expert_input_param_idx = moe_experts._expert_input_param_idx;
        m_config.compiled_models = moe_experts._compiled_models;

        // Validate configuration
        if (!m_config.is_valid()) {
            OPENVINO_THROW("Invalid MoE configuration");
        }

        LOG_DEBUG("MoE Config: num_experts=" << m_config.num_experts
                                             << ", num_active_experts=" << m_config.num_active_experts
                                             << ", input_token_count=" << m_config.input_token_count
                                             << ", mode=" << (m_config.is_decoding() ? "DECODING" : "PREFILL"));

        // Step 2: Initialize shared resources (only once)
        LOG_DEBUG("Initializing shared MoE resources...");
        m_resources.initialize_shared(m_config, m_allocator, get_device_name(idx, &desc));
    } else {
        LOG_DEBUG("Reusing existing shared MoE config and resources");
    }

    // Step 3: Initialize request cache and create request pool (for decoding mode only)
    if (pool_size > 0 && m_config.is_decoding()) {
        LOG_DEBUG("Creating request cache pool for sublayer[" << idx << "] with pool_size=" << pool_size);

        // Initialize cache structure (creates RequestCache on first call)
        m_resources.initialize_cache(num_sublayers, pool_size);

        // Create request pool for this sublayer
        std::vector<RqPtr> requests;
        requests.resize(pool_size);

        // Create first request separately (needed for tensor sharing)
        try {
            requests[0] = desc.compiled_model->create_infer_request();
            requests[0]->infer();  // Warmup
            LOG_DEBUG("Created and warmed up request[0]");
        } catch (const std::exception& ex) {
            LOG_ERROR("Failed to create MoE pool request[0] for sublayer[" << idx << "]: " << ex.what());
            throw;
        }

        // Create remaining requests in parallel, sharing tensors from first request
        ov::parallel_for(pool_size - 1, [&](size_t i) {
            const size_t req_idx = i + 1;
            try {
                auto request = desc.compiled_model->create_infer_request();

                // Share all input & output tensors from first request to save memory
                const auto& inputs = desc.compiled_model->inputs();
                for (size_t input_idx = 0; input_idx < inputs.size(); ++input_idx) {
                    request->set_tensor(inputs[input_idx], requests[0]->get_tensor(inputs[input_idx]));
                }
                const auto& outputs = desc.compiled_model->outputs();
                for (size_t output_idx = 0; output_idx < outputs.size(); ++output_idx) {
                    request->set_tensor(outputs[output_idx], requests[0]->get_tensor(outputs[output_idx]));
                }

                request->infer();  // Warmup
                requests[req_idx] = std::move(request);
            } catch (const std::exception& ex) {
                LOG_ERROR("Failed to create MoE pool request[" << req_idx << "] for sublayer[" << idx
                                                               << "]: " << ex.what());
                throw;
            }
        });

        // Initialize cache layer with pre-allocated requests
        m_resources.request_cache->initialize_layer(idx, std::move(requests));
        LOG_DEBUG("Request pool created with " << pool_size << " requests");
    } else {
        LOG_DEBUG("Request cache disabled (pool_size=" << pool_size << ", is_decoding=" << m_config.is_decoding()
                                                       << ")");
    }

    LOG_INFO("MoE preparation completed for sublayer[" << idx << "]");
}

void MoEExecutor::run(size_t real_idx,
                      size_t idx,
                      const MoEIO& io,
                      std::map<size_t, std::vector<size_t>>& token_to_experts,
                      std::map<size_t, std::vector<size_t>>& expert_to_tokens) {
    // Validate config is initialized
    if (!m_config.is_valid()) {
        OPENVINO_THROW("MoEExecutor::run() - Configuration not initialized");
    }

    LOG_DEBUG("\n========== MoE Expert Inference [Subgraph " << idx << "] ==========");

    const auto num_experts = m_config.num_experts;
    const auto num_active_experts = m_config.num_active_experts;
    const auto input_token_count = m_config.input_token_count;
    const bool is_decoding = (input_token_count == 1);

    LOG_DEBUG("MoE Config: num_experts=" << num_experts << ", num_active_experts=" << num_active_experts
                                         << ", input_token_count=" << input_token_count
                                         << ", mode=" << (is_decoding ? "DECODING" : "PREFILL"));

    if (!io.router_scores) {
        OPENVINO_THROW("MoE: Router scores are required but not available");
    }

    // Parse router scores and populate routing maps
    // Note: parse_selected_experts_from_router() clears the maps internally before populating
    std::vector<size_t> selected_experts;
    if (is_decoding) {
        m_profiler("decoding", "Parse Router Output", [&]() {
            selected_experts = ov::npuw::moe::parse_selected_experts_from_router(io.router_scores,
                                                                                 num_experts,
                                                                                 token_to_experts,
                                                                                 expert_to_tokens);
        });
    } else {
        m_profiler("prefill", "Parse Router Output", [&]() {
            selected_experts = ov::npuw::moe::parse_selected_experts_from_router(io.router_scores,
                                                                                 num_experts,
                                                                                 token_to_experts,
                                                                                 expert_to_tokens);
        });
    }

    if (selected_experts.empty()) {
        OPENVINO_THROW("MoE: No experts selected by router");
    }

    // Dispatch to appropriate inference function
    if (is_decoding) {
        m_profiler("decoding", "Total Decoding", [&]() {
            run_batch_experts(idx, real_idx, selected_experts, io);
        });
    } else {
        m_profiler("prefill", "Total Prefill", [&]() {
            run_iterative_experts(idx, real_idx, selected_experts, io, token_to_experts, expert_to_tokens);
        });
    }

    LOG_DEBUG("========== MoE Expert Inference Completed ==========");
}

void MoEExecutor::recreate_requests(size_t real_idx) {
    LOG_INFO("Recreating MoE requests for submodel[" << real_idx << "]...");

    // Reset entire request cache (all sublayers will need re-initialization)
    m_resources.reset_cache();

    LOG_INFO("MoE requests recreated for submodel[" << real_idx << "]");
}

void MoEExecutor::run_batch_experts(size_t idx,
                                    size_t real_idx,
                                    const std::vector<size_t>& selected_experts,
                                    const MoEIO& io) {
    LOG_DEBUG("\n[BATCH EXPERTS] Processing single token with " << selected_experts.size() << " experts in parallel");

    const auto num_active_experts = m_config.num_active_experts;

    // Validate expert count
    if (selected_experts.size() != num_active_experts) {
        OPENVINO_THROW("MoE Batch experts: number of selected experts does not match num_active_experts");
    }

    // Step 1: Try to find cached request (O(1) lookup) - if cache is enabled
    // Note: Use idx because each function call has its own cache pool
    RqPtr request{};
    size_t pool_idx = 0;

    const bool cache_enabled = (m_resources.request_cache != nullptr);

    if (cache_enabled) {
        request = m_resources.request_cache->find(idx, selected_experts);
    }

    if (!request) {
        // Cache MISS or cache disabled: Get idle/LRU request or create new
        if (cache_enabled) {
            auto [idle_request, idx_in_pool] = m_resources.request_cache->get_idle_or_lru(idx);
            request = idle_request;
            pool_idx = idx_in_pool;
        } else {
            // Cache disabled - use the regular subrequest
            request = m_accessor.get_subrequest(real_idx);
        }

        // Step 2: Configure expert weights
        m_profiler("decoding", "Unpack Closure", [&]() {
            m_accessor.unpack_multiple_experts_closure(idx, request, selected_experts);
        });

        // Step 3: Register to cache for future hits (only if cache enabled)
        if (cache_enabled) {
            m_resources.request_cache->register_request(idx, pool_idx, selected_experts);
        }
    }

    // Get compiled model descriptor
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    // Step 4: Bind input/output tensors to cached request (must bind every time as these tensors change)
    // 4.1: Bind expert input (token embeddings)
    if (m_config.expert_input_param_idx.has_value()) {
        const auto expert_input_idx = m_config.expert_input_param_idx.value();
        const auto& expert_input_port = desc.compiled_model->inputs()[expert_input_idx];
        auto expert_input_tensor = io.expert_input;
        if (!expert_input_tensor) {
            OPENVINO_THROW("MoE: Expert input tensor not available");
        }
        request->set_tensor(expert_input_port, expert_input_tensor);
    }

    // 4.2: Bind output tensor (from io, set in function_prologue)
    const auto& output_port = desc.compiled_model->outputs()[0];
    auto output_tensor = io.outputs.at(0);
    if (!output_tensor) {
        OPENVINO_THROW("MoE: Output tensor not available");
    }
    request->set_tensor(output_port, output_tensor);

    // Step 5: Set unrolled router scores (always needed, even on cache hit)
    m_profiler("decoding", "Set Router Input", [&]() {
        set_router_scores(idx, real_idx, selected_experts, request, io);
    });

    // Step 6: Execute inference once for all K experts in parallel
    m_profiler("decoding", "Expert Inference", [&]() {
        request->infer();
    });
}

void MoEExecutor::run_iterative_experts(size_t idx,
                                        size_t real_idx,
                                        const std::vector<size_t>& selected_experts,
                                        const MoEIO& io,
                                        std::map<size_t, std::vector<size_t>>& token_to_experts,
                                        std::map<size_t, std::vector<size_t>>& expert_to_tokens) {
    LOG_DEBUG("\n[ITERATIVE EXPERTS] Processing multiple tokens by iterating through experts");

    const auto input_token_count = m_config.input_token_count;

    // Clear output buffer before accumulating expert outputs
    if (!m_resources.expert_output_accumulator) {
        OPENVINO_THROW("MoE: Expert output accumulator is null");
    }
    std::memset(m_resources.expert_output_accumulator->data(),
                0,
                m_resources.expert_output_accumulator->get_byte_size());

    // Get tensor references (constant across all experts)
    if (!m_config.router_scores_idx.has_value()) {
        OPENVINO_THROW("MoE: Router scores index not available");
    }
    if (!m_config.expert_input_param_idx.has_value()) {
        OPENVINO_THROW("MoE: Expert input parameter index not available");
    }

    auto router_source = io.router_scores;
    auto expert_input_source = io.expert_input;

    // Calculate output embedding dimension from any compiled model (all have same output shape)
    auto any_compiled_model = m_config.compiled_models.begin()->second;
    auto output_shape = any_compiled_model->outputs()[0].get_shape();
    size_t embed_dim = (output_shape.size() == 4) ? output_shape[3] : output_shape[1];

    // Get compiled model descriptor for accessing infer requests
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    // Process each expert sequentially
    for (size_t expert_id : selected_experts) {
        LOG_DEBUG("\n  Processing Expert[" << expert_id << "]...");

        // Get tokens assigned to this expert
        const auto& tokens_for_expert = expert_to_tokens.at(expert_id);
        const size_t total_tokens = tokens_for_expert.size();

        LOG_DEBUG("    Expert[" << expert_id << "] processing " << total_tokens << " tokens");

        // Precompute expert slot for each token
        // This eliminates map lookup + linear search in scatter_expert_outputs hot loop
        std::vector<size_t> expert_slots_for_tokens(total_tokens);
        for (size_t i = 0; i < total_tokens; ++i) {
            size_t token_id = tokens_for_expert[i];
            const auto& expert_ids = token_to_experts.at(token_id);
            auto it = std::find(expert_ids.begin(), expert_ids.end(), expert_id);
            if (it == expert_ids.end()) {
                OPENVINO_THROW("MoE: Token should have this expert");
            }
            expert_slots_for_tokens[i] = std::distance(expert_ids.begin(), it);
        }

        // Process tokens in dynamic chunks based on available compiled models
        // Priority: use largest possible chunk_size (256 > 128 > 64 > 32)
        size_t processed_tokens = 0;
        size_t last_chunk_size = 0;  // Track last used chunk_size to detect changes
                                     // Note: When switching to a new expert, last_chunk_size resets to 0,
                                     // ensuring weights are unpacked for the first chunk of each expert
        while (processed_tokens < total_tokens) {
            size_t remaining_tokens = total_tokens - processed_tokens;

            // Chunk selection strategy:
            // - remaining_tokens <= smallest chunk → use smallest chunk
            // - remaining_tokens >= largest chunk → use largest chunk
            // - otherwise → use smallest chunk that is >= remaining_tokens
            // Pre-sorted in descending order: [256, 128, 64, 32, 16]
            if (m_resources.sorted_chunk_sizes.empty()) {
                OPENVINO_THROW("MoE: Sorted chunk sizes cannot be empty");
            }

            size_t selected_chunk_size;
            size_t smallest_chunk = m_resources.sorted_chunk_sizes.back();
            size_t largest_chunk = m_resources.sorted_chunk_sizes.front();

            if (remaining_tokens <= smallest_chunk) {
                // Use smallest chunk
                selected_chunk_size = smallest_chunk;
            } else if (remaining_tokens >= largest_chunk) {
                // Use largest chunk
                selected_chunk_size = largest_chunk;
            } else {
                // Find smallest chunk >= remaining_tokens
                // Since sorted descending, iterate from end to find first >= remaining_tokens
                selected_chunk_size = smallest_chunk;  // Default to smallest
                for (auto it = m_resources.sorted_chunk_sizes.rbegin(); it != m_resources.sorted_chunk_sizes.rend();
                     ++it) {
                    if (*it >= remaining_tokens) {
                        selected_chunk_size = *it;
                        break;
                    }
                }
            }

            // Actual tokens to process in this iteration
            size_t current_chunk_size = std::min(selected_chunk_size, remaining_tokens);

            LOG_DEBUG("      Processing tokens [" << processed_tokens << ", " << (processed_tokens + current_chunk_size)
                                                  << ") with chunk_size=" << selected_chunk_size);

            // Get selected infer request and compiled model from CompiledModelDesc
            auto infer_request_it = desc.moe_infer_requests.find(selected_chunk_size);
            if (infer_request_it == desc.moe_infer_requests.end()) {
                OPENVINO_THROW("MoE: Infer request for chunk_size=", selected_chunk_size, " not found");
            }
            auto selected_infer_request = infer_request_it->second;

            // Get input/output ports for selected infer request
            auto selected_compiled_model = m_config.compiled_models.at(selected_chunk_size);
            const auto& selected_router_iport = selected_compiled_model->inputs()[m_config.router_scores_idx.value()];
            const auto& selected_expert_input_iport =
                selected_compiled_model->inputs()[m_config.expert_input_param_idx.value()];
            const auto& selected_oport = selected_compiled_model->outputs()[0];

            auto selected_router_dest = selected_infer_request->get_tensor(selected_router_iport);
            auto selected_expert_input_dest = selected_infer_request->get_tensor(selected_expert_input_iport);
            auto selected_expert_output = selected_infer_request->get_tensor(selected_oport);

            // Step 1: Unpack expert weights when chunk_size changes (different infer request)
            // This ensures each infer request has correct weights loaded
            // Important: last_chunk_size is reset to 0 at the start of each expert loop,
            // so the first chunk of a new expert will always trigger unpacking (0 != selected_chunk_size)
            if (selected_chunk_size != last_chunk_size) {
                m_profiler("prefill", "Unpack Closure", [&]() {
                    m_accessor.unpack_single_expert_closure(idx, selected_infer_request, expert_id);
                });
                last_chunk_size = selected_chunk_size;
            }

            // Step 2: Gather router scores for this chunk
            m_profiler("prefill", "Gather Router Scores", [&]() {
                ov::npuw::moe::gather_router_scores(router_source,
                                                    selected_router_dest,
                                                    expert_id,
                                                    tokens_for_expert,
                                                    processed_tokens,
                                                    current_chunk_size);
            });

            // Step 3: Gather expert inputs for this chunk
            m_profiler("prefill", "Gather Expert Input", [&]() {
                ov::npuw::moe::gather_expert_inputs(expert_input_source,
                                                    selected_expert_input_dest,
                                                    tokens_for_expert,
                                                    processed_tokens,
                                                    current_chunk_size);
            });

            // Step 4: Execute expert inference
            m_profiler("prefill", "Expert Inference", [&]() {
                selected_infer_request->infer();
            });

            // Step 5: Scatter expert outputs back to global buffer
            m_profiler("prefill", "Scatter Output", [&]() {
                ov::npuw::moe::scatter_expert_outputs(selected_expert_output,
                                                      m_resources.expert_output_accumulator,
                                                      tokens_for_expert,
                                                      processed_tokens,
                                                      current_chunk_size,
                                                      embed_dim,
                                                      input_token_count,
                                                      expert_slots_for_tokens);
            });

            // Move to next chunk
            processed_tokens += current_chunk_size;
        }
        LOG_DEBUG("    Expert[" << expert_id << "] completed");
    }
}

void MoEExecutor::set_router_scores(size_t idx,
                                    size_t real_idx,
                                    const std::vector<size_t>& selected_experts,
                                    RqPtr& request,
                                    const MoEIO& io) {
    const auto num_active_experts = m_config.num_active_experts;

    // Validate router scores index is provided
    if (!m_config.router_scores_idx.has_value()) {
        OPENVINO_THROW("MoE: Router input parameter index not specified for expert model");
    }

    const auto original_router_idx = m_config.router_scores_idx.value();
    const auto& param_mapping = m_config.param_mapping;

    // Find unrolled router score parameters
    auto mapping_it = param_mapping.find(original_router_idx);
    if (mapping_it == param_mapping.end()) {
        LOG_WARN("Router parameter index " << original_router_idx << " not found in param_mapping");
        OPENVINO_THROW("MoE: Router parameter not in mapping - unexpected for decoding mode");
    }

    const auto& unrolled_router_indices = mapping_it->second;
    LOG_DEBUG("  Setting " << unrolled_router_indices.size() << " router score parameters");

    // Validate unrolled parameter count
    if (unrolled_router_indices.size() != num_active_experts) {
        LOG_ERROR("Router parameter count mismatch: expected " << num_active_experts << ", got "
                                                               << unrolled_router_indices.size());
        OPENVINO_THROW("MoE: Router parameter unroll count mismatch");
    }

    // Get router scores source tensor and element type
    auto router_scores_source = io.router_scores;
    auto elem_type = router_scores_source->get_element_type();

    // Get compiled model from config
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    // Set each unrolled router score parameter
    for (size_t k = 0; k < num_active_experts; ++k) {
        size_t expert_id = selected_experts[k];
        size_t unrolled_param_idx = unrolled_router_indices[k];

        const auto& router_iport = desc.compiled_model->inputs()[unrolled_param_idx];
        auto router_tensor = request->get_tensor(router_iport);

        // Copy router score from source[expert_id] to dest[0]
        if (elem_type == ov::element::f16) {
            auto* src = router_scores_source->data<ov::float16>();
            auto* dst = router_tensor->data<ov::float16>();
            dst[0] = src[expert_id];
        } else if (elem_type == ov::element::f32) {
            auto* src = router_scores_source->data<float>();
            auto* dst = router_tensor->data<float>();
            dst[0] = src[expert_id];
        } else {
            OPENVINO_THROW("Unsupported router scores element type: ", elem_type);
        }
    }
}

std::string MoEExecutor::get_device_name(size_t idx, const void* compiled_model_desc_ptr) const {
    const auto& desc = compiled_model_desc_ptr
                           ? *static_cast<const CompiledModel::CompiledModelDesc*>(compiled_model_desc_ptr)
                           : *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(idx));
    return *desc.device_it;
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
