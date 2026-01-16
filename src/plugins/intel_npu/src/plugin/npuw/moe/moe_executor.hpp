// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../util.hpp"  // For TensorPtr
#include "moe_config.hpp"
#include "moe_resources.hpp"
#include "moe_types.hpp"  // For MoEIO
#include "openvino/runtime/so_ptr.hpp"

// Forward declarations
namespace ov {
class ITensor;
namespace npuw {
class CompiledModel;
}  // namespace npuw
}  // namespace ov

namespace ov {
namespace npuw {

// Forward declare CompiledModelDesc to avoid circular dependency
// Will be included in implementation file
namespace moe {
class MoEExecutor;
}

// Import TensorPtr from util namespace for use in this header
using ov::npuw::util::TensorPtr;

/**
 * @brief Accessor interface for MoEExecutor to access JustInferRequest internals
 *
 * This interface decouples MoEExecutor from JustInferRequest implementation details,
 * following the Dependency Inversion Principle. JustInferRequest implements this
 * interface to provide controlled access to its internal resources.
 */
class ISubrequestAccessor {
public:
    virtual ~ISubrequestAccessor() = default;

    /**
     * @brief Get subrequest by index
     * @param idx Submodel index
     * @return Infer request pointer
     */
    virtual ov::SoPtr<ov::IAsyncInferRequest> get_subrequest(size_t idx) = 0;

    /**
     * @brief Get compiled submodel descriptor
     * @param idx Submodel index
     * @return Reference to CompiledModelDesc
     */
    virtual const void* get_submodel_desc(size_t idx) = 0;  // Returns CompiledModelDesc*

    /**
     * @brief Allocate memory tensor
     * @param type Element type
     * @param shape Tensor shape
     * @param device Target device
     * @return Allocated tensor pointer
     */
    virtual TensorPtr allocate_mem(const ov::element::Type& type,
                                   const ov::Shape& shape,
                                   const std::string& device) = 0;

    /**
     * @brief Unpack function closure (non-expert parameters)
     * @param idx Function call index
     * @param request Target infer request
     */
    virtual void unpack_closure(size_t idx, ov::SoPtr<ov::IAsyncInferRequest> request) = 0;

    /**
     * @brief Unpack single expert's closure (expert-specific weights)
     * @param idx Function call index
     * @param request Target infer request
     * @param expert_id Expert ID to unpack
     */
    virtual void unpack_single_expert_closure(size_t idx,
                                              ov::SoPtr<ov::IAsyncInferRequest> request,
                                              size_t expert_id) = 0;

    /**
     * @brief Unpack multiple experts' closure (batch expert mode)
     * @param idx Function call index
     * @param request Target infer request
     * @param expert_ids Expert IDs to unpack
     */
    virtual void unpack_multiple_experts_closure(size_t idx,
                                                 ov::SoPtr<ov::IAsyncInferRequest> request,
                                                 const std::vector<size_t>& expert_ids) = 0;
};

namespace moe {

/**
 * @brief MoE execution engine
 *
 * Encapsulates all MoE-specific inference logic, isolated from JustInferRequest.
 * Manages expert routing, weight unpacking, caching, and execution for both
 * decoding (single token) and prefill (multiple tokens) modes.
 *
 * Design principles:
 * - Composition over inheritance: Used as a component in JustInferRequest
 * - Dependency injection: Accesses host request via ISubrequestAccessor
 * - Single responsibility: Only handles MoE inference logic
 */
class MoEExecutor {
public:
    using RqPtr = ov::SoPtr<ov::IAsyncInferRequest>;
    using AllocatorFn = std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)>;
    using ProfilerFn = std::function<void(const std::string& area, const std::string& name, std::function<void()> fn)>;

    /**
     * @brief Constructor
     *
     * @param accessor Interface to access JustInferRequest internals
     * @param profiler Profiling callback (area, metric name, function)
     * @param allocator Memory allocation function
     */
    MoEExecutor(ISubrequestAccessor& accessor, ProfilerFn profiler, AllocatorFn allocator);

    /**
     * @brief Prepare MoE resources for a sublayer
     *
     * Must be called once for each sublayer (including function calls) before first run.
     * Initializes:
     * - Config: Once on first call (singleton, shared by all sublayers)
     * - Shared resources: Once on first call (sorted_chunk_sizes, expert_output_accumulator)
     * - Request cache: Created on first call (manages all sublayers)
     *
     * @param idx Sublayer index (used for cache layer indexing, may be function call)
     * @param real_idx Real function body index (for accessing config descriptor)
     * @param num_sublayers Total number of sublayers (for cache creation on first call)
     * @param pool_size Request cache pool size (0 = disabled)
     */
    void prepare(size_t idx, size_t real_idx, size_t num_sublayers, size_t pool_size);

    /**
     * @brief Execute MoE inference
     *
     * Main entry point called from JustInferRequest::unsafe_infer().
     * Handles router parsing, expert selection, and dispatches to
     * appropriate execution mode (batch vs iterative).
     *
     * @param real_idx Submodel index (after replaced_by resolution)
     * @param idx Function call index
     * @param io MoE I/O tensors (router_scores, expert_input, outputs)
     * @param token_to_experts Routing map: token_id -> [expert_ids] (reusable storage)
     * @param expert_to_tokens Routing map: expert_id -> [token_ids] (reusable storage)
     */
    void run(size_t real_idx,
             size_t idx,
             const MoEIO& io,
             std::map<size_t, std::vector<size_t>>& token_to_experts,
             std::map<size_t, std::vector<size_t>>& expert_to_tokens);

    /**
     * @brief Get expert output accumulator buffer
     *
     * Returns the shared output buffer used for accumulating expert outputs
     * in prefill mode. This buffer is allocated during prepare() and reused
     * across all inference calls.
     *
     * @return Pointer to expert output accumulator tensor
     */
    TensorPtr get_output_accumulator() const {
        return m_resources.expert_output_accumulator;
    }

private:
    // === Dependency injection ===
    ISubrequestAccessor& m_accessor;  // Access to JustInferRequest internals
    ProfilerFn m_profiler;            // Performance profiling callback
    AllocatorFn m_allocator;          // Memory allocation function

    // === State management ===
    // MoE configuration (single instance, shared by all sublayers)
    // Sanity check in JustInferRequest ensures only one MoE type exists
    MoEConfig m_config;

    // Shared resources for all sublayers
    // - request_caches: per-sublayer (map inside MoEResources, indexed by idx)
    // - sorted_chunk_sizes: shared (single instance)
    // - expert_output_accumulator: shared (single instance)
    MoEResources m_resources;  // Single instance, shared by all sublayers

    // === Execution modes ===

    /**
     * @brief Execute batch experts inference (decoding mode)
     *
     * Processes one token with K active experts in parallel.
     * Uses cached infer requests for specific expert combinations.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     * @param io MoE I/O tensors
     */
    void run_batch_experts(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts, const MoEIO& io);

    /**
     * @brief Execute iterative experts inference (prefill mode)
     *
     * Processes multiple tokens by iterating through experts sequentially.
     * Uses dynamic chunk sizing and accumulates outputs.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     * @param io MoE I/O tensors
     * @param token_to_experts Routing map: token_id -> [expert_ids]
     * @param expert_to_tokens Routing map: expert_id -> [token_ids]
     */
    void run_iterative_experts(size_t idx,
                               size_t real_idx,
                               const std::vector<size_t>& selected_experts,
                               const MoEIO& io,
                               std::map<size_t, std::vector<size_t>>& token_to_experts,
                               std::map<size_t, std::vector<size_t>>& expert_to_tokens);

    // === Helper functions ===

    /**
     * @brief Set unrolled router scores for batch expert mode
     *
     * Maps router scores from [num_experts] to K unrolled parameters.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     * @param request Target infer request
     * @param io MoE I/O tensors (contains router_scores)
     */
    void set_router_scores(size_t idx,
                           size_t real_idx,
                           const std::vector<size_t>& selected_experts,
                           RqPtr& request,
                           const MoEIO& io);

    /**
     * @brief Get device name for a submodel
     *
     * @param idx Submodel/function call index
     * @param compiled_model_desc Optional compiled model descriptor (used during prepare)
     * @return Device name string
     */
    std::string get_device_name(size_t idx, const void* compiled_model_desc = nullptr) const;
};

}  // namespace moe
}  // namespace npuw
}  // namespace ov
