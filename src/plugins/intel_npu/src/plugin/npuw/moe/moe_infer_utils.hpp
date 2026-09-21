// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

#include "../perf.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace npuw {
namespace moe {

/**
 * @brief Single source of truth for MoE profiling bucket names.
 *
 * Chunk-size-dependent tags (NPU Wait, Scatter Output) are NOT here: their
 * names depend on runtime chunk size values, so they're precomputed into
 * MoEResources::wait_tag / scatter_tag instead of being static literals.
 */
namespace tags {
constexpr const char* kParseRouterOutput = "Parse Router Output";
constexpr const char* kTotalExpertBatch = "Total Expert Batch";
constexpr const char* kTotalExpertIterative = "Total Expert Iterative";
constexpr const char* kUnpackClosure = "Unpack Closure";
constexpr const char* kSetRouterInput = "Set Router Input";
constexpr const char* kExpertInference = "Expert Inference";
constexpr const char* kGetOutputTensor = "Get Output Tensor";
constexpr const char* kParseRouterRow = "Parse Router Row";
constexpr const char* kGetIOTensors = "Get I/O Tensors";
constexpr const char* kGatherRouterScores = "Gather Router Scores";
constexpr const char* kGatherExpertInput = "Gather Expert Input";
constexpr const char* kNpuStart = "NPU Start";
constexpr const char* kClearUnfilledSlots = "Clear Unfilled Slots";
}  // namespace tags

/**
 * @brief MoE performance profiling structure.
 *
 * Tracks performance metrics for MoE operations in different processing modes.
 * Uses ov::npuw::perf::Profile framework for automatic metric collection and reporting.
 */
struct MoEProfile {
    ov::npuw::perf::Profile<ov::npuw::perf::metric<ov::npuw::perf::MSec>> iterative;  // EXPERT_ITERATIVE mode
    ov::npuw::perf::Profile<ov::npuw::perf::metric<ov::npuw::perf::MSec>> batch;      // EXPERT_BATCH mode

    MoEProfile();
};

/**
 * @brief Standalone stats for EXPERT_ITERATIVE token routing and chunk padding.
 *
 * Independent of MoEProfile (which only times named code blocks): this tracks
 * *what* gets dispatched rather than *how long* it takes — per-expert token
 * counts (routing skew) and per-chunk-size padding waste (real tokens vs the
 * fixed chunk capacity actually paid for on the NPU).
 */
struct MoEChunkStats {
    // Accumulated across all run_expert_iterative() calls for one expert_id.
    struct PerExpert {
        uint64_t selections = 0;  // number of calls where this expert got >=1 token
        uint64_t tokens = 0;      // total tokens routed to this expert
        uint64_t chunks = 0;      // total chunk dispatches used to process them
    };

    // Accumulated across all chunks dispatched at a given compiled chunk size.
    struct PerChunkSize {
        uint64_t chunks = 0;  // number of chunks dispatched at this size
        uint64_t tokens = 0;  // real (non-padding) tokens carried by those chunks
    };

    // Upper-exclusive bucket edges for the "remaining tokens" histogram, sampled right
    // before each select_chunk() call. Locates real gaps between compiled chunk sizes
    // (e.g. a spike at 128-192 means a cs=192 model would help), independent of
    // whatever chunk sizes happen to be compiled today.
    static constexpr std::array<size_t, 9> remaining_hist_edges = {8, 16, 32, 64, 128, 192, 256, 384, 512};

    bool enabled = false;
    std::vector<PerExpert> per_expert;             // indexed by expert_id
    std::map<size_t, PerChunkSize> per_chunk_size;  // keyed by chunk size
    std::array<uint64_t, remaining_hist_edges.size() + 1> remaining_hist{};  // last bucket = ">= last edge"

    // Must be called once num_experts is known (prepare()), before any record_*() call.
    void init(size_t num_experts, bool active);

    void record_expert(size_t expert_id, uint64_t tokens, uint64_t chunks);
    void record_chunk(size_t chunk_size, uint64_t actual_tokens);
    void record_remaining(size_t remaining);

    void report() const;

    ~MoEChunkStats() {
        if (enabled) {
            try {
                report();
            } catch (...) {
            }
        }
    }
};

template <typename T>
inline bool is_nonzero(T v) {
    return std::abs(static_cast<float>(v)) > 1e-6f;
}

/**
 * @brief Slice a single expert's weight from batched weight tensor.
 *
 * Creates a zero-copy view of expert weight from batched tensor with shape [num_experts, ...].
 * Handles both sub-byte (4-bit) and regular (>=8-bit) element types.
 *
 * @param batched_weight Source tensor with shape [num_experts, ...]
 * @param expert_id ID of the expert to slice (0 to num_experts-1)
 * @param num_experts Total number of experts in the batch
 * @return ov::Tensor Zero-copy view with shape [1, ...] for the selected expert
 */
ov::Tensor slice_expert_weight(const ov::Tensor& batched_weight, size_t expert_id, size_t num_experts);

/**
 * @brief Parse selected experts from router output tensor.
 *
 * Analyzes router output to determine which experts are selected by each token.
 * Router output has shape [num_experts, 1, token_num, 1] with non-zero weights
 * indicating expert selection.
 *
 * @param router_output Router output tensor with expert selection weights
 * @param num_experts Total number of experts
 * @param token_to_experts[out] Map from token_id to list of selected expert_ids
 * @param expert_to_tokens[out] Map from expert_id to list of tokens that selected it
 * @return Vector of selected expert IDs (sorted)
 */
std::vector<size_t> parse_selected_experts_from_router(const ov::SoPtr<ov::ITensor>& router_output,
                                                       size_t num_experts,
                                                       std::map<size_t, std::vector<size_t>>& token_to_experts,
                                                       std::map<size_t, std::vector<size_t>>& expert_to_tokens);

/**
 * @brief Optimized tensor setter for small/large tensors.
 *
 * Uses copy for small tensors (<= 11520 bytes) to avoid set_tensor overhead.
 * Uses set_tensor for large tensors (zero-copy).
 *
 * @param request Inference request
 * @param iport Input port to set tensor for
 * @param tensor_impl Tensor to set or copy
 */
void set_tensor_optimized(ov::SoPtr<ov::IAsyncInferRequest> request,
                          const ov::Output<const ov::Node>& iport,
                          const ov::SoPtr<ov::ITensor>& tensor_impl);

/**
 * @brief Gather router scores for selected tokens from router output.
 *
 * Extracts router scores for a specific expert and token range from batched router output.
 * Supports both 4D [num_experts, 1, token_num, 1] and 2D [num_experts, token_num] shapes.
 *
 * @param router_source Source router tensor with all expert scores
 * @param router_dest Destination tensor to write gathered scores
 * @param expert_id ID of the expert to gather scores for
 * @param token_ids List of token IDs to process
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens to process
 */
void gather_router_scores(const ov::SoPtr<ov::ITensor>& router_source,
                          const ov::SoPtr<ov::ITensor>& router_dest,
                          size_t expert_id,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size);

/**
 * @brief Gather expert input embeddings for selected tokens.
 *
 * Extracts input embeddings for a range of tokens from the source tensor.
 * Supports both 2D [num_tokens, hidden_dim] and 4D [1, 1, num_tokens, hidden_dim] shapes.
 *
 * @param input_source Source tensor with all token embeddings
 * @param input_dest Destination tensor to write gathered embeddings
 * @param token_ids List of token IDs to process
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens to process
 */
void gather_expert_inputs(const ov::SoPtr<ov::ITensor>& input_source,
                          const ov::SoPtr<ov::ITensor>& input_dest,
                          const std::vector<size_t>& token_ids,
                          size_t chunk_start,
                          size_t chunk_size);

/**
 * @brief Scatter expert outputs back to global output buffer.
 *
 * Writes expert output for processed tokens to the correct positions in the global
 * output buffer based on expert slot assignments.
 *
 * @param expert_output Expert's output tensor for current chunk
 * @param global_output_buffer Global output buffer [K, 1, num_tokens, embed_dim]
 * @param token_ids List of token IDs that were processed
 * @param chunk_start Starting index in token_ids list
 * @param chunk_size Number of tokens processed
 * @param embed_dim Embedding dimension size
 * @param input_token_count Total number of input tokens
 * @param expert_slots_for_tokens Precomputed expert slot for each token in token_ids (same size as token_ids)
 */
void scatter_expert_outputs(const ov::SoPtr<ov::ITensor>& expert_output,
                            const ov::SoPtr<ov::ITensor>& global_output_buffer,
                            const std::vector<size_t>& token_ids,
                            size_t chunk_start,
                            size_t chunk_size,
                            size_t embed_dim,
                            size_t input_token_count,
                            const std::vector<size_t>& expert_slots_for_tokens);

/**
 * @brief Zero out only the (token, expert_slot) cells that scatter_expert_outputs() never wrote.
 *
 * Cheaper than clearing the whole accumulator up front: with dense top-K routing every
 * token normally fills all num_active_experts slots, so this is a near-free O(num_tokens)
 * scan in the common case, touching memory only for the rare token whose is_nonzero()
 * filtering left a slot unfilled.
 *
 * @param global_output_buffer Global output buffer [K, 1, num_tokens, embed_dim]
 * @param token_slot_count Per-token count of slots actually filled during this call
 * @param num_active_experts K, the number of expert slots expected per token
 * @param embed_dim Embedding dimension size
 * @param input_token_count Total number of input tokens
 */
void clear_unfilled_accumulator_slots(const ov::SoPtr<ov::ITensor>& global_output_buffer,
                                      const std::vector<size_t>& token_slot_count,
                                      size_t num_active_experts,
                                      size_t embed_dim,
                                      size_t input_token_count);

/**
 * @brief MoE Request Cache - LRU cache for expert inference requests.
 *
 * Manages a fixed-size pool of pre-configured inference requests per MoE sublayer.
 * Caches requests by expert combination to avoid redundant weight configuration.
 * Uses LRU eviction when pool is full.
 */
class RequestCache {
public:
    using RqPtr = ov::SoPtr<ov::IAsyncInferRequest>;

    /**
     * @brief Construct a new Request Cache.
     *
     * @param num_layers Total number of sublayers (indexed 0 to num_layers-1)
     * @param pool_size_per_layer Number of cached requests per MoE layer
     */
    RequestCache(size_t num_layers, size_t pool_size_per_layer);

    /**
     * @brief Destructor - prints cache statistics on destruction.
     */
    ~RequestCache();

    /**
     * @brief Find cached request for expert combination.
     *
     * @param sublayer_idx Sublayer index
     * @param expert_ids Expert IDs (order preserved, not sorted)
     * @return RqPtr Cached request if found, nullptr on cache miss
     */
    RqPtr find(size_t sublayer_idx, const std::vector<size_t>& expert_ids);

    /**
     * @brief Get idle or least-recently-used request from pool.
     *
     * Returns an unconfigured request from free list (if available),
     * or evicts the LRU entry and returns it for reconfiguration.
     *
     * @param sublayer_idx Sublayer index
     * @return std::pair<RqPtr, size_t> {request, pool_index}
     */
    std::pair<RqPtr, size_t> get_idle_or_lru(size_t sublayer_idx);

    /**
     * @brief Register configured request to cache.
     *
     * Associates the given pool entry with the expert combination.
     * Removes entry from free list and adds to LRU index.
     *
     * @param sublayer_idx Sublayer index
     * @param pool_idx Index in the pool
     * @param expert_ids Expert IDs this request is configured for
     */
    void register_request(size_t sublayer_idx, size_t pool_idx, const std::vector<size_t>& expert_ids);

    /**
     * @brief Initialize request pool for a specific MoE sublayer.
     *
     * Pre-allocates inference requests for the given sublayer.
     * Must be called before using find/get_idle_or_lru for this sublayer.
     *
     * @param sublayer_idx Sublayer index
     * @param requests Pre-allocated inference requests for this layer
     */
    void initialize_layer(size_t sublayer_idx, std::vector<RqPtr>&& requests);

    /**
     * @brief Get cache hit rate statistics.
     *
     * @return std::pair<uint64_t, uint64_t> {total_queries, cache_hits}
     */
    std::pair<uint64_t, uint64_t> get_statistics() const;

    /**
     * @brief Print cache statistics to log.
     */
    void print_statistics() const;

private:
    struct PoolEntry {
        RqPtr request;                   // Inference request
        std::vector<size_t> expert_ids;  // Expert IDs currently configured
        uint64_t last_use_iter;          // LRU timestamp
        bool is_configured;              // Whether request has valid configuration
    };

    size_t m_pool_size_per_layer;

    // Pool storage: indexed by sublayer_idx
    std::vector<std::vector<PoolEntry>> m_pool;

    // Fast lookup: sublayer_idx -> (expert_ids_key -> pool_idx)
    std::vector<std::unordered_map<std::string, size_t>> m_cache_lookup;

    // Free list: sublayer_idx -> list of unconfigured pool indices
    std::vector<std::vector<size_t>> m_free_list;

    // LRU index: sublayer_idx -> sorted set of (last_use_iter, pool_idx)
    std::vector<std::set<std::pair<uint64_t, size_t>>> m_lru_index;

    // Statistics
    bool m_report_on_die = false;

    mutable uint64_t m_total_queries = 0;
    mutable uint64_t m_cache_hits = 0;

    // Helper: Convert expert_ids to cache key string
    std::string make_cache_key(const std::vector<size_t>& expert_ids) const;
};

}  // namespace moe
}  // namespace npuw
}  // namespace ov
