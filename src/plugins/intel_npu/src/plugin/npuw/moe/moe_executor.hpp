// Copyright (C) 2018-2026 Intel Corporation
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
     * @brief Check if closure parameter is a gather closure (should be skipped)
     * @param idx Submodel index
     * @param cidx Closure parameter index
     * @return True if it's a gather closure
     */
    virtual bool is_gather_closure(size_t idx, size_t cidx) = 0;

    /**
     * @brief Check if unpacking is required for closure parameter
     * @param idx Submodel index
     * @param cidx Closure parameter index
     * @return True if unpacking is required
     */
    virtual bool unpack_required(size_t idx, size_t cidx) = 0;

    /**
     * @brief Check if tensor copy is needed for closure parameter
     * @param idx Submodel index
     * @param cidx Closure parameter index
     * @return True if copy is needed
     */
    virtual bool needs_copy_closure(size_t idx, size_t cidx) = 0;
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

    /**
     * @brief Constructor
     *
     * @param accessor Interface to access JustInferRequest internals
     * @param allocator Memory allocation function
     */
    MoEExecutor(ISubrequestAccessor& accessor, AllocatorFn allocator);

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
     */
    void run(size_t real_idx, size_t idx);

    /**
     * @brief Handle MoE input parameter binding in function_prologue
     *
     * Called for each input parameter in function_prologue loop.
     * Handles router_scores, expert_input, and downstream layer inputs.
     *
     * @param idx Function call index
     * @param real_idx Real function body index
     * @param param_idx Input parameter index
     * @param i_tensor Input tensor
     * @return true if this parameter was handled by MoE (skip default binding)
     */
    bool function_prologue_moe_input(size_t idx,
                                     size_t real_idx,
                                     size_t param_idx,
                                     const ov::SoPtr<ov::ITensor>& i_tensor);

    /**
     * @brief Handle MoE output binding in function_prologue
     *
     * Called for each output in function_prologue loop.
     * Stores output tensor in MoE I/O structure for later use.
     *
     * @param idx Function call index
     * @param output_idx Output index
     * @param o_tensor Output tensor
     * @return true if this output was handled by MoE (skip default binding)
     */
    bool function_prologue_moe_output(size_t idx, size_t output_idx, const ov::SoPtr<ov::ITensor>& o_tensor);

    /**
     * @brief Get MoE performance profile statistics
     *
     * @return Optional reference to MoE profile (nullopt if profiling disabled)
     */
    const std::optional<MoEProfile>& get_profile() const {
        return m_profile;
    }

private:
    // === Dependency injection ===
    ISubrequestAccessor& m_accessor;  // Access to JustInferRequest internals
    AllocatorFn m_allocator;          // Memory allocation function

    // === Profiling ===
    std::optional<MoEProfile>
        m_profile;  // Performance statistics (always collects, reports based on profiling_enabled())

    // === Weight unpacking methods ===
    /**
     * @brief Unpack single expert's closure (expert-specific weights)
     * @param idx Function call index
     * @param request Target infer request
     * @param expert_id Expert ID to unpack
     */
    void unpack_single_expert_closure(size_t idx, RqPtr request, size_t expert_id);

    /**
     * @brief Unpack multiple experts' closure (batch expert mode)
     * @param idx Function call index
     * @param request Target infer request
     * @param expert_ids Expert IDs to unpack
     */
    void unpack_multiple_experts_closure(size_t idx, RqPtr request, const std::vector<size_t>& expert_ids);

    // === State management ===
    // MoE configuration (single instance, shared by all sublayers)
    // Sanity check in JustInferRequest ensures only one MoE type exists
    MoEConfig m_config;

    // Shared resources for all sublayers
    // - request_caches: per-sublayer (map inside MoEResources, indexed by idx)
    // - sorted_chunk_sizes: shared (single instance)
    // - expert_output_accumulator: shared (single instance)
    MoEResources m_resources;  // Single instance, shared by all sublayers

    // Routing maps (reused across inferences to avoid repeated allocation)
    std::map<size_t, std::vector<size_t>> m_token_to_experts;
    std::map<size_t, std::vector<size_t>> m_expert_to_tokens;

    // MoE I/O storage (per-sublayer)
    std::vector<MoEIO> m_moe_io;

    // === Execution modes ===

    /**
     * @brief Execute expert batch mode inference
     *
     * Processing mode: Single token processed with K experts in parallel
     * Uses cached infer requests for specific expert combinations.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     */
    void run_expert_batch(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts);

    /**
     * @brief Execute expert iterative mode inference
     *
     * Processing mode: Iterate through experts, each processes multiple tokens
     * Uses dynamic chunk sizing and accumulates outputs.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     */
    void run_expert_iterative(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts);

    /**
     * @brief Execute batch experts inference (deprecated)
     * @deprecated Use run_expert_batch() instead
     */
    [[deprecated("Use run_expert_batch() instead")]]
    void run_batch_experts(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts) {
        run_expert_batch(idx, real_idx, selected_experts);
    }

    /**
     * @brief Execute iterative experts inference (deprecated)
     * @deprecated Use run_expert_iterative() instead
     */
    [[deprecated("Use run_expert_iterative() instead")]]
    void run_iterative_experts(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts) {
        run_expert_iterative(idx, real_idx, selected_experts);
    }

    // === Helper functions ===

    /**
     * @brief Set unrolled router scores for expert batch mode
     *
     * Maps router scores from [num_experts] to K unrolled parameters.
     *
     * @param idx Function call index
     * @param real_idx Submodel index
     * @param selected_experts List of selected expert IDs
     * @param request Target infer request
     */
    void set_router_scores(size_t idx, size_t real_idx, const std::vector<size_t>& selected_experts, RqPtr& request);

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
