// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_executor.hpp"

#include <optional>

#include "../compiled_model.hpp"  // For CompiledModel::CompiledModelDesc
#include "../logging.hpp"
#include "moe_infer_utils.hpp"
#include "moe_types.hpp"  // For MoEIO definition
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"  // For ov::parallel_for

namespace ov {
namespace npuw {
namespace moe {

MoEExecutor::MoEExecutor(ISubrequestAccessor& accessor, AllocatorFn allocator)
    : m_accessor(accessor),
      m_allocator(std::move(allocator)) {
    // Profile will be initialized using profiling_enabled() in MoEProfile constructor
    m_profile.emplace();  // MoEProfile constructor reads profiling_enabled()
    LOG_DEBUG("MoEExecutor created");
}

void MoEExecutor::prepare(size_t idx, size_t real_idx, size_t num_sublayers, size_t pool_size) {
    LOG_INFO("Preparing MoE resources for sublayer[" << idx << "] (real_idx=" << real_idx << ")...");
    LOG_BLOCK();

    // Get function body descriptor directly (no need to resolve, caller already did it)
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    const auto* moe_experts = get_compiled_experts(desc.pipeline.context);
    if (moe_experts == nullptr) {
        OPENVINO_THROW("MoEExecutor::prepare called on non-MoE submodel[real_idx=", real_idx, "]");
    }

    // Step 1: Initialize MoE config (only once, on first prepare call)
    if (!m_config.is_valid()) {
        LOG_DEBUG("Creating shared MoE config");

        m_config.num_experts = moe_experts->num_experts;
        m_config.num_active_experts = moe_experts->num_active_experts;
        m_config.input_token_count = moe_experts->input_token_count;
        m_config.expert_hidden_dim = moe_experts->expert_hidden_dim;
        m_config.param_mapping = moe_experts->_param_mapping;
        m_config.router_scores.original = moe_experts->_router_scores.original;
        m_config.router_scores.compiled = moe_experts->_router_scores.compiled;
        m_config.expert_input.original = moe_experts->_expert_input.original;
        m_config.expert_input.compiled = moe_experts->_expert_input.compiled;
        m_config.compiled_models = moe_experts->_compiled_models;

        // Validate configuration
        if (!m_config.is_valid()) {
            OPENVINO_THROW("Invalid MoE configuration");
        }

        LOG_DEBUG("MoE Config: num_experts=" << m_config.num_experts
                                             << ", num_active_experts=" << m_config.num_active_experts
                                             << ", input_token_count=" << m_config.input_token_count
                                             << ", mode=" << get_mode_name(m_config.get_processing_mode()));

        // Step 2: Initialize resources based on processing mode
        LOG_DEBUG("Initializing MoE resources...");
        if (m_config.is_expert_batch_mode()) {
            m_resources.initialize_expert_batch_mode(m_config,
                                                     m_allocator,
                                                     get_device_name(idx, &desc),
                                                     num_sublayers,
                                                     pool_size);
        } else {
            m_resources.initialize_expert_iterative_mode(m_config, m_allocator, get_device_name(idx, &desc));
        }
    } else {
        LOG_DEBUG("Reusing existing shared MoE config and resources");
    }

    // Step 3: Initialize request pool for this sublayer (expert batch mode only)
    if (pool_size > 0 && m_config.is_expert_batch_mode()) {
        std::vector<RqPtr> requests;
        requests.resize(pool_size);

        // Create first request separately (needed for tensor sharing)
        requests[0] = desc.compiled_model->create_infer_request();
        requests[0]->infer();  // Warmup
        LOG_DEBUG("Created and warmed up request[0]");

        // Create remaining requests in parallel, sharing tensors from first request
        ov::parallel_for(pool_size - 1, [&](size_t i) {
            const size_t req_idx = i + 1;
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
        });

        // Initialize cache layer with pre-allocated requests
        NPUW_ASSERT(m_resources.request_cache && "Request cache must be initialized for batch mode with pool_size > 0");
        m_resources.request_cache->initialize_layer(idx, std::move(requests));
        LOG_DEBUG("Request pool created with " << pool_size << " requests");
    } else {
        LOG_DEBUG("Request cache disabled (pool_size=" << pool_size << ", mode="
                                                       << get_mode_name(m_config.get_processing_mode()) << ")");
    }

    // Step 4: Initialize MoE I/O structure for this sublayer
    if (m_moe_io.empty()) {
        m_moe_io.resize(num_sublayers);
        LOG_DEBUG("Pre-allocated MoE I/O storage for " << num_sublayers << " sublayers");
    }
    NPUW_ASSERT(idx < m_moe_io.size() && "Sublayer index out of range");

    const auto num_outputs = desc.compiled_model->outputs().size();
    m_moe_io[idx].outputs.resize(num_outputs);
    LOG_DEBUG("Initialized MoE I/O with " << num_outputs << " outputs");

    LOG_INFO("MoE preparation completed for sublayer[" << idx << "]");
}

bool MoEExecutor::function_prologue_moe_input(size_t idx,
                                              size_t real_idx,
                                              size_t param_idx,
                                              const ov::SoPtr<ov::ITensor>& i_tensor) {
    // Get descriptor
    const auto& desc =
        *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    // MoE expert layer: handle router scores and expert inputs
    if (const auto* moe = get_compiled_experts(desc.pipeline.context)) {
        // Router scores: store for later use in MoE inference
        if (moe->_router_scores.original.has_value() && param_idx == moe->_router_scores.original.value()) {
            NPUW_ASSERT(idx < m_moe_io.size() && "MoEExecutor::prepare() must be called first");
            m_moe_io[idx].router_scores = i_tensor;
            return true;  // Handled by MoE
        }

        // Expert inputs: store for later use (batch mode: cache binding, iterative mode: chunking)
        if (moe->_expert_input.original.has_value() && param_idx == moe->_expert_input.original.value()) {
            NPUW_ASSERT(idx < m_moe_io.size() && "MoEExecutor::prepare() must be called first");
            m_moe_io[idx].expert_input = i_tensor;
            return true;  // Handled by MoE
        }

        // For expert layer, all param_base parameters should be router_scores or expert_input
        NPUW_ASSERT(false && "Unknown MoE expert parameter index");
        return false;  // Unreachable
    }

    // MoE downstream layer: bind expert output accumulator or other parameters
    if (const auto* moe_downstream = get_compiled_downstream(desc.pipeline.context)) {
        const auto& iport = desc.compiled_model->inputs()[param_idx];
        auto real_request = m_accessor.get_subrequest(real_idx);

        if (param_idx == moe_downstream->expert_output_param_idx) {
            // Bind expert output accumulator
            auto output_buffer = m_resources.expert_output_accumulator;
            NPUW_ASSERT(output_buffer && "MoE output buffer not available");
            real_request->set_tensor(iport, output_buffer);
        } else {
            // Other parameters use the input tensor directly
            real_request->set_tensor(iport, i_tensor);
        }
        return true;  // All downstream layer parameters handled by MoE
    }

    // Should not reach here if is_moe flag is set correctly
    NPUW_ASSERT(false && "MoE flag set but no compiled MoE state found");
    return false;
}

bool MoEExecutor::function_prologue_moe_output(size_t idx, size_t output_idx, const ov::SoPtr<ov::ITensor>& o_tensor) {
    NPUW_ASSERT(idx < m_moe_io.size() && "MoEExecutor::prepare() must be called first");

    // Store output tensor in MoE I/O structure
    // For EXPERT_BATCH mode: deferred for cache lookup
    // For EXPERT_ITERATIVE mode: relayouted from expert outputs, expert_output_accumulator will be used eventually
    m_moe_io[idx].outputs.at(output_idx) = o_tensor;
    return true;  // Always handled by MoE for expert layers
}

void MoEExecutor::run(size_t real_idx, size_t idx) {
    // Validate config is initialized
    if (!m_config.is_valid()) {
        OPENVINO_THROW("MoEExecutor::run() - Configuration not initialized");
    }

    LOG_DEBUG("\n========== MoE Expert Inference [Subgraph " << idx << "] ==========");

    const auto num_experts = m_config.num_experts;
    const auto num_active_experts = m_config.num_active_experts;
    const auto input_token_count = m_config.input_token_count;
    const auto processing_mode = m_config.get_processing_mode();

    LOG_DEBUG("MoE Config: num_experts=" << num_experts << ", num_active_experts=" << num_active_experts
                                         << ", input_token_count=" << input_token_count
                                         << ", mode=" << get_mode_name(processing_mode));

    // Get I/O for this sublayer
    const auto& io = m_moe_io[idx];

    if (!io.router_scores) {
        OPENVINO_THROW("MoE: Router scores are required but not available");
    }

    // Dispatch to appropriate inference function.
    // EXPERT_BATCH: parse upfront (single token, routing maps needed for cache lookup).
    // EXPERT_ITERATIVE: parse is fused inside run_expert_iterative, expert-row by row,
    //   overlapping with NPU execution to hide the per-layer parse overhead.
    if (processing_mode == MoEProcessingMode::EXPERT_BATCH) {
        std::vector<size_t> selected_experts;
        m_profile->batch["Parse Router Output"].record([&]() {
            selected_experts = ov::npuw::moe::parse_selected_experts_from_router(io.router_scores,
                                                                                 num_experts,
                                                                                 m_token_to_experts,
                                                                                 m_expert_to_tokens);
        });
        if (selected_experts.empty()) {
            OPENVINO_THROW("MoE: No experts selected by router");
        }
        m_profile->batch["Total Expert Batch"].record([&]() {
            run_expert_batch(idx, real_idx, selected_experts);
        });
    } else {
        m_profile->iterative["Total Expert Iterative"].record([&]() {
            run_expert_iterative(idx);
        });
    }

    LOG_DEBUG("========== MoE Expert Inference Completed ==========");
}

void MoEExecutor::run_expert_batch(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts) {
    LOG_DEBUG("\n[EXPERT_BATCH] Processing single token with " << selected_experts.size() << " experts in parallel");

    const auto num_active_experts = m_config.num_active_experts;
    const auto& io = m_moe_io[idx];

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
        m_profile->batch["Unpack Closure"].record([&]() {
            unpack_multiple_experts_closure(idx, request, selected_experts);
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
    // Use the transformed parameter index to access the compiled model's port correctly.
    // (unroll_expert_dimension may shift the expert_input parameter index relative to the original model)
    if (m_config.expert_input.compiled.has_value()) {
        const auto expert_input_idx = m_config.expert_input.compiled.value();
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
    m_profile->batch["Set Router Input"].record([&]() {
        set_router_scores(idx, real_idx, selected_experts, request);
    });

    // Step 6: Execute inference once for all K experts in parallel
    m_profile->batch["Expert Inference"].record([&]() {
        request->infer();
    });
}

void MoEExecutor::run_expert_iterative(size_t idx) {
    LOG_DEBUG("\n[EXPERT_ITERATIVE] Processing multiple tokens with async inference and overlapping NPU execution");

    const auto& io = m_moe_io[idx];

    // Precondition checks
    if (!m_resources.expert_output_accumulator)
        OPENVINO_THROW("MoE: Expert output accumulator is null");
    if (!io.expert_input)
        OPENVINO_THROW("MoE: Expert input tensor is not set — prologue must bind it before run()");
    if (!m_config.router_scores.compiled.has_value())
        OPENVINO_THROW("MoE: Router scores index not available");
    if (!m_config.expert_input.compiled.has_value())
        OPENVINO_THROW("MoE: Expert input parameter index not available");
    if (m_resources.sorted_chunk_sizes.empty())
        OPENVINO_THROW("MoE: Sorted chunk sizes cannot be empty");

    // Clear output accumulator before accumulating expert outputs
    std::memset(m_resources.expert_output_accumulator->data(),
                0,
                m_resources.expert_output_accumulator->get_byte_size());

    auto expert_input_source = io.expert_input;

    // Output embedding dimension
    const auto output_shape = m_config.compiled_models.begin()->second->outputs()[0].get_shape();
    const size_t embed_dim = (output_shape.size() == 4) ? output_shape[3] : output_shape[1];

    // num_tokens and num_experts come from config, validated once during prepare().
    // The router tensor's token dimension is guaranteed to match input_token_count because
    // both are derived from the same compiled model structure at prepare() time.
    const size_t num_tokens = m_config.input_token_count;
    const size_t num_experts = m_config.num_experts;

    // Chunk-size selector
    auto select_chunk = [&](size_t remaining) -> size_t {
        const size_t smallest = m_resources.sorted_chunk_sizes.back();
        const size_t largest = m_resources.sorted_chunk_sizes.front();
        if (remaining <= smallest)
            return smallest;
        if (remaining >= largest)
            return largest;
        for (auto it = m_resources.sorted_chunk_sizes.rbegin(); it != m_resources.sorted_chunk_sizes.rend(); ++it) {
            if (*it >= remaining)
                return *it;
        }
        return smallest;
    };

    // Double-buffer request helpers
    size_t global_slot = 0;
    auto get_req = [&](size_t cs, size_t slot) -> RqPtr& {
        return m_resources.chunk_infer_requests.at(cs)[slot & 1];
    };
    // Tracks which expert_id is currently loaded into each (chunk_size, slot) request.
    // Key = cs * 2 + (slot & 1): unique because all chunk_sizes are distinct integers,
    // so cs1 * 2 + b1 == cs2 * 2 + b2 only when cs1 == cs2 and b1 == b2.
    std::map<size_t, size_t> req_expert_state;
    auto do_unpack = [&](size_t cs, size_t slot, RqPtr& req, size_t expert_id) {
        const size_t key = cs * 2 + (slot & 1);
        auto it = req_expert_state.find(key);
        if (it == req_expert_state.end() || it->second != expert_id) {
            m_profile->iterative["Unpack Closure"].record([&]() {
                unpack_single_expert_closure(idx, req, expert_id);
            });
            req_expert_state[key] = expert_id;
        }
    };

    // Expert data ring buffer:
    // 2 slots alternate per non-empty expert.  At most one slot is "in-flight" for
    // scatter at any time; the other slot is being filled for the next expert.
    // Safety: expert[k+2] reuses the slot that expert[k] used, and by then expert[k]'s
    // last work item has already been drained (pipeline depth = 1).
    struct ExpertData {
        std::vector<size_t> tokens;
        std::vector<size_t> slots;
    };
    std::array<ExpertData, 2> expert_ring;
    size_t expert_ring_idx = 0;  // incremented per non-empty expert

    // In-flight item tracking
    struct InflightItem {
        size_t cs;
        size_t req_slot;
        size_t chunk_start;
        size_t chunk_tokens;
        size_t ring_idx;  // expert_ring slot index at launch time
    };
    std::optional<InflightItem> inflight;

    // Drain the in-flight item referenced by `inflight` (wait + scatter).
    // Called both inside the pipeline loop (to drain the previous item while NPU
    // runs the current one) and once after the loop to drain the final item.
    auto do_drain = [&]() {
        NPUW_ASSERT(inflight.has_value() && "do_drain called with no in-flight item");
        auto& req = get_req(inflight->cs, inflight->req_slot);
        m_profile->iterative["NPU Wait"].record([&]() {
            req->wait();
        });
        const auto& data = expert_ring[inflight->ring_idx & 1];
        const auto cm = m_config.compiled_models.at(inflight->cs);
        auto output = req->get_tensor(cm->outputs()[0]);
        m_profile->iterative["Scatter Output"].record([&]() {
            ov::npuw::moe::scatter_expert_outputs(output,
                                                  m_resources.expert_output_accumulator,
                                                  data.tokens,
                                                  inflight->chunk_start,
                                                  inflight->chunk_tokens,
                                                  embed_dim,
                                                  num_tokens,
                                                  data.slots);
        });
    };

    // Per-token counter: how many earlier experts have already selected this token.
    // Gives each token's expert-slot index in O(1) without a full two-pass scan.
    std::vector<size_t> token_slot_count(num_tokens, 0);

    // Parse-ahead buffer: pre-scan the NEXT expert's router row after drain(k-1) but
    // while NPU k is still executing.  This hides the O(num_tokens) threshold scan
    // inside the NPU overlap window instead of paying for it on the critical path
    // before item k+1's dispatch.
    //
    // The buffer stores only token IDs that passed the threshold (v > 1e-6).
    // Slot assignment (O(selected) not O(num_tokens)) is still done at the
    // top of the next expert iteration.
    //
    // Timeline with parse-ahead:
    //   CPU: [Unpack(k)+Gather(k)] → start(k) → drain(k-1) → [Parse(k+1)] → [Unpack(k+1)+Gather(k+1)] → start(k+1)
    //   NPU:                         [=================NPU(k)==================] [==NPU(k+1)==]
    //
    struct ParseAheadBuffer {
        std::vector<size_t> tokens;   // pre-filtered token IDs for next expert
        size_t expert_id = SIZE_MAX;  // which expert was pre-scanned (sentinel = none)
    };
    ParseAheadBuffer parse_ahead;

    // Per-expert loop: three phases — Parse, Dispatch, Prefetch.
    auto stream_and_run = [&](auto* data) {
        for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
            // -- Parse: determine which tokens this expert processes --
            const size_t ring_slot = expert_ring_idx & 1;
            auto& cur = expert_ring[ring_slot];
            cur.tokens.clear();
            cur.slots.clear();

            const auto* row = data + expert_id * num_tokens;
            // Shared by both the parse-ahead fast path and the full threshold scan below.
            auto fill_token = [&](size_t token_id) {
                cur.tokens.push_back(token_id);
                cur.slots.push_back(token_slot_count[token_id]++);
            };
            m_profile->iterative["Parse Router Row"].record([&]() {
                if (parse_ahead.expert_id == expert_id) {
                    // Fast path: token IDs already filtered; only slot updates remain.
                    for (size_t token_id : parse_ahead.tokens) {
                        fill_token(token_id);
                    }
                    parse_ahead.expert_id = SIZE_MAX;  // consumed
                } else {
                    // Full O(num_tokens) threshold scan (first expert, or parse-ahead missed).
                    for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
                        if (is_nonzero(row[token_id])) {
                            fill_token(token_id);
                        }
                    }
                }
            });

            if (cur.tokens.empty()) {
                continue;  // expert not selected — do not advance ring_idx
            }

            // -- Dispatch: slice tokens into chunks and pipeline NPU execution --
            size_t processed = 0;
            while (processed < cur.tokens.size()) {
                const size_t cs = select_chunk(cur.tokens.size() - processed);
                const size_t actual = std::min(cs, cur.tokens.size() - processed);
                const size_t s = global_slot & 1;
                auto& req = get_req(cs, s);

                do_unpack(cs, s, req, expert_id);
                {
                    const auto cm = m_config.compiled_models.at(cs);
                    auto router_dest = req->get_tensor(cm->inputs()[m_config.router_scores.compiled.value()]);
                    auto input_dest = req->get_tensor(cm->inputs()[m_config.expert_input.compiled.value()]);
                    m_profile->iterative["Gather Router Scores"].record([&]() {
                        ov::npuw::moe::gather_router_scores(io.router_scores,
                                                            router_dest,
                                                            expert_id,
                                                            cur.tokens,
                                                            processed,
                                                            actual);
                    });
                    m_profile->iterative["Gather Expert Input"].record([&]() {
                        ov::npuw::moe::gather_expert_inputs(expert_input_source,
                                                            input_dest,
                                                            cur.tokens,
                                                            processed,
                                                            actual);
                    });
                }

                // Start NPU first, then drain previous — this creates the CPU/NPU overlap.
                m_profile->iterative["NPU Start"].record([&]() {
                    req->start_async();
                });
                if (inflight) {
                    do_drain();
                }

                inflight = InflightItem{cs, s, processed, actual, ring_slot};
                ++global_slot;
                processed += actual;
            }

            // -- Prefetch: scan next expert's row while the last NPU chunk runs --
            if ((expert_id + 1) < num_experts) {
                const size_t next_id = expert_id + 1;
                const auto* next_row = data + next_id * num_tokens;
                parse_ahead.tokens.clear();
                parse_ahead.expert_id = next_id;
                for (size_t token_id = 0; token_id < num_tokens; ++token_id) {
                    if (is_nonzero(next_row[token_id])) {
                        parse_ahead.tokens.push_back(token_id);
                    }
                }
            }

            ++expert_ring_idx;
        }
    };

    const auto elem_type = io.router_scores->get_element_type();
    if (elem_type == ov::element::f32) {
        stream_and_run(io.router_scores->data<float>());
    } else if (elem_type == ov::element::f16) {
        stream_and_run(io.router_scores->data<ov::float16>());
    } else {
        OPENVINO_THROW("MoE: Unsupported router element type for iterative inference");
    }

    if (!inflight) {
        OPENVINO_THROW("MoE: No experts selected by router");
    }

    // Drain the last in-flight item
    do_drain();
}

void MoEExecutor::set_router_scores(size_t idx,
                                    size_t real_idx,
                                    const std::vector<size_t>& selected_experts,
                                    RqPtr& request) {
    const auto num_active_experts = m_config.num_active_experts;
    const auto& io = m_moe_io[idx];

    // Validate router scores index is provided
    if (!m_config.router_scores.original.has_value()) {
        OPENVINO_THROW("MoE: Router input parameter index not specified for expert model");
    }

    const auto original_router_idx = m_config.router_scores.original.value();
    const auto& param_mapping = m_config.param_mapping;

    // Find unrolled router score parameters
    auto mapping_it = param_mapping.find(original_router_idx);
    if (mapping_it == param_mapping.end()) {
        LOG_WARN("Router parameter index " << original_router_idx << " not found in param_mapping");
        OPENVINO_THROW("MoE: Router parameter not in mapping - unexpected for expert batch mode");
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

std::string MoEExecutor::get_device_name(size_t idx, const void*) const {
    return m_accessor.subgraph_device(idx);
}

void MoEExecutor::unpack_single_expert_closure(std::size_t idx, RqPtr request, size_t expert_id) {
    // Get model descriptors
    const auto& comp_model_desc =
        *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(idx));
    NPUW_ASSERT(comp_model_desc.replaced_by);

    const auto real_idx = comp_model_desc.replaced_by.value();
    const auto& func_desc =
        *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    const auto* moe_experts = get_compiled_experts(func_desc.pipeline.context);
    NPUW_ASSERT(moe_experts != nullptr);
    const auto num_experts = moe_experts->num_experts;

    auto& desc_closure = comp_model_desc.closure.get().closure;

    for (std::size_t cidx = 0u; cidx < desc_closure.size(); cidx++) {
        auto& closure = desc_closure[cidx];
        const auto closure_param_id = comp_model_desc.param_base + cidx;

        // Skip gather closures
        if (m_accessor.is_gather_closure(idx, cidx)) {
            continue;
        }

        auto& iport = func_desc.compiled_model->inputs()[closure_param_id];

        // Check if this weight needs slicing (has num_experts in first dimension)
        auto closure_shape = closure.get_shape();
        bool needs_slicing = !closure_shape.empty() && closure_shape[0] == num_experts;

        if (needs_slicing) {
            // Slice expert weight using view (no copy) - returns ov::Tensor object
            auto sliced_weight_tensor = ov::npuw::moe::slice_expert_weight(closure, expert_id, num_experts);

            // Get impl pointer for use in unpacking/setting
            auto sliced_weight = ov::get_tensor_impl(sliced_weight_tensor);

            // Handle unpacking if needed
            if (m_accessor.unpack_required(idx, cidx)) {
                auto clparam = request->get_tensor(iport);

                if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx] && comp_model_desc.zerops[cidx]) {
                    ov::npuw::util::unpack(sliced_weight,
                                           ov::get_tensor_impl(comp_model_desc.zerops[cidx]),
                                           ov::get_tensor_impl(comp_model_desc.scales[cidx]),
                                           clparam);
                } else if (!comp_model_desc.scales.empty() && comp_model_desc.scales[cidx]) {
                    ov::npuw::util::unpack(sliced_weight, ov::get_tensor_impl(comp_model_desc.scales[cidx]), clparam);
                } else {
                    ov::npuw::util::unpack(sliced_weight, clparam);
                }
            } else {
                // Direct set (no unpacking needed)
                // When cache is enabled: Use direct set_tensor to avoid polluting shared input tensors
                // When cache is disabled: Use set_tensor_optimized for better performance (copies small tensors)
                if (m_resources.request_cache) {
                    request->set_tensor(iport, sliced_weight);
                } else {
                    ov::npuw::moe::set_tensor_optimized(request, iport, sliced_weight);
                }
            }
        } else {
            // This closure parameter doesn't need slicing, use original logic
            if (m_accessor.needs_copy_closure(idx, cidx)) {
                auto clparam = request->get_tensor(iport);
                ov::get_tensor_impl(closure)->copy_to(clparam._ptr);
            } else {
                request->set_tensor(iport, ov::get_tensor_impl(closure));
            }
        }
    }
}

void MoEExecutor::unpack_multiple_experts_closure(std::size_t idx,
                                                  RqPtr request,
                                                  const std::vector<size_t>& expert_ids) {
    /**
     * MoE Batch Expert Closure Unpacking for Expert Batch Mode
     *
     * Purpose: Process K active experts simultaneously (batch mode, 1 token)
     *
     * Input:
     *   - Batched closure parameters: shape [num_experts, ...] (original model)
     *   - expert_ids: K selected expert IDs [e0, e1, e2, e3]
     *
     * Output:
     *   - Unrolled model parameters populated with sliced expert weights
     *   - Each parameter receives data for one specific expert
     *
     * Flow:
     *   1. Iterate over original closure parameters
     *   2. For each closure parameter, process its K unrolled variants
     *   3. Slice expert weight from batched tensor
     *   4. Unpack (if dtype mismatch) or direct set (if dtype match)
     */

    // ========== Step 1: Get model descriptors and validate MoE structure ==========
    const auto& comp_model_desc =
        *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(idx));
    NPUW_ASSERT(comp_model_desc.replaced_by);

    const auto real_idx = comp_model_desc.replaced_by.value();
    const auto& func_desc =
        *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));

    const auto* moe_experts = get_compiled_experts(func_desc.pipeline.context);
    NPUW_ASSERT(moe_experts != nullptr);
    const auto num_experts = moe_experts->num_experts;        // Total experts in model (e.g., 32)
    const size_t K = expert_ids.size();                       // Active experts (e.g., 4)
    const auto& param_mapping = moe_experts->_param_mapping;  // original_idx -> [unrolled_indices]

    auto& desc_closure = comp_model_desc.closure.get().closure;
    const auto& compiled_inputs = func_desc.compiled_model->inputs();

    // ========== Step 2: Process each original closure parameter ==========
    // Iterate over closure parameters, then process their K unrolled variants
    for (std::size_t closure_idx = 0; closure_idx < desc_closure.size(); ++closure_idx) {
        // Skip gather closures (dummy tensors, not real weights)
        if (m_accessor.is_gather_closure(idx, closure_idx)) {
            continue;
        }

        // Calculate original parameter index in the model
        const size_t original_param_idx = comp_model_desc.param_base + closure_idx;

        // Check if this parameter has unrolled variants (is in param_mapping)
        auto mapping_it = param_mapping.find(original_param_idx);
        if (mapping_it == param_mapping.end()) {
            continue;  // Not unrolled
        }

        const auto& unrolled_indices = mapping_it->second;
        NPUW_ASSERT(unrolled_indices.size() == K);

        auto& batched_closure = desc_closure[closure_idx];
        const auto& closure_shape = batched_closure.get_shape();

        // Verify this is a batched parameter [num_experts, ...]
        const bool is_batched = !closure_shape.empty() && closure_shape[0] == num_experts;
        if (!is_batched) {
            // Shared parameter path - all K unrolled parameters share same weight
            const bool do_copy = m_accessor.needs_copy_closure(idx, closure_idx);
            auto batched_impl = ov::get_tensor_impl(batched_closure);

            for (size_t position = 0; position < K; ++position) {
                const auto& iport = compiled_inputs[unrolled_indices[position]];
                if (do_copy) {
                    batched_impl->copy_to(request->get_tensor(iport)._ptr);
                } else {
                    request->set_tensor(iport, batched_impl);
                }
            }
            continue;
        }

        // ========== Step 3: Process batched parameters (K experts) ==========

        // Pre-determine unpack configuration (same for all K experts)
        const auto batched_elem_type = batched_closure.get_element_type();
        const auto target_elem_type = request->get_tensor(compiled_inputs[unrolled_indices[0]])->get_element_type();
        const bool needs_unpack = (batched_elem_type != target_elem_type);

        ov::SoPtr<ov::ITensor> scales_impl, zerops_impl;
        if (needs_unpack) {
            if (!comp_model_desc.scales.empty() && comp_model_desc.scales[closure_idx]) {
                scales_impl = ov::get_tensor_impl(comp_model_desc.scales[closure_idx]);
            }
            if (!comp_model_desc.zerops.empty() && comp_model_desc.zerops[closure_idx]) {
                zerops_impl = ov::get_tensor_impl(comp_model_desc.zerops[closure_idx]);
            }
        }

        // Process K experts
        for (size_t position = 0; position < K; ++position) {
            const size_t expert_id = expert_ids[position];
            const auto& iport = compiled_inputs[unrolled_indices[position]];

            // Slice expert weight (zero-copy view)
            ov::Tensor sliced_expert = ov::npuw::moe::slice_expert_weight(batched_closure, expert_id, num_experts);

            if (needs_unpack) {
                // Unpack path (dtype mismatch)
                auto sliced_impl = ov::get_tensor_impl(sliced_expert);
                auto clparam = request->get_tensor(iport);

                if (scales_impl && zerops_impl) {
                    ov::npuw::util::unpack(sliced_impl, zerops_impl, scales_impl, clparam);
                } else if (scales_impl) {
                    ov::npuw::util::unpack(sliced_impl, scales_impl, clparam);
                } else if (zerops_impl) {
                    ov::npuw::util::unpack(sliced_impl, zerops_impl, clparam);
                } else {
                    ov::npuw::util::unpack(sliced_impl, clparam);
                }
            } else {
                auto sliced_impl = ov::get_tensor_impl(sliced_expert);
                // Direct set (no unpacking needed)
                // When cache is enabled: Use direct set_tensor to avoid polluting shared input tensors
                // When cache is disabled: Use set_tensor_optimized for better performance (copies small tensors)
                if (m_resources.request_cache) {
                    request->set_tensor(iport, sliced_impl);
                } else {
                    ov::npuw::moe::set_tensor_optimized(request, iport, sliced_impl);
                }
            }
        }  // for each expert
    }  // for each closure parameter

    LOG_DEBUG("Done unpacking MoE batch expert closure");
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
