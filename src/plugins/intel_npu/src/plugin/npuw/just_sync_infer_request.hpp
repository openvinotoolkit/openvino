// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "base_sync_infer_request.hpp"
#include "host_flash_attention.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "spatial.hpp"

namespace ov {
namespace npuw {

class CompiledModel;
class AsyncInferRequest;

class MemAccessSim {
public:
    explicit MemAccessSim(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    using ReadList = std::list<LinkFrom>;
    const ReadList& read_list(std::size_t idx) const;

    std::size_t remaining_reads(const LinkFrom& from);
    void register_read(const LinkFrom& from);

private:
    std::map<LinkFrom, std::size_t> m_remaining_reads;
    std::vector<ReadList> m_read_list;
};

class FuncMemMgr {
    MemAccessSim m_sim;
    std::shared_ptr<ov::npuw::CompiledModel> m_model;

    void assign(const LinkFrom& from);

    // Function ID -> Output port number
    using FO = std::pair<std::size_t, std::size_t>;
    struct Assignment {
        TensorPtr ptr;
        LinkFrom from;
    };
    std::map<FO, std::vector<Assignment>> m_memory;  // Dynamic assignment table
    std::map<LinkFrom, TensorPtr> m_table;           // Static allocation/assignment table

public:
    explicit FuncMemMgr(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    using AllocFcn = std::function<TensorPtr(const ov::element::Type&, const ov::Shape&, const std::string&)>;
    void set_alloc(AllocFcn&& fcn);
    void assign_memory();

    TensorPtr get_tensor(const LinkFrom& from);

private:
    AllocFcn m_alloc;
};

class JustInferRequest final : public IBaseInferRequest {
public:
    explicit JustInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);
    ~JustInferRequest();

protected:
    ////////////////////////////////////
    // implement IBaseInferRequest
    void prepare_for_infer() override;
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t idx) override;
    void run_subrequest_for_success(std::size_t idx, bool& failover) override;
    void subscribe_subrequest(std::size_t idx, Completed cb) override;
    void complete_subrequest(std::size_t idx) override;
    void cancel_subrequest(std::size_t idx) override;
    bool supports_async_pipeline() const override;
    void update_subrequest_links(std::size_t idx) override;

    TensorPtr alloc_global_out(std::size_t out_idx) const override;

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

    ////////////////////////////////////
    // now own API

    // FIXME: probably this one should go to the base class too
    RqPtr get_real_subrequest(std::size_t idx);

    void bind_global_parameters(std::size_t idx);
    void bind_global_results(std::size_t idx);
    using IBaseInferRequest::bind_global_results;

    void function_prologue(std::size_t idx);
    void function_prologue_attn(std::size_t real_idx, std::size_t idx);
    void function_prologue_pyramid_attn(std::size_t real_idx, std::size_t idx);

    void unsafe_during(std::size_t real_idx, std::size_t idx, const std::function<void()>& f);
    void unsafe_infer(std::size_t real_idx, std::size_t idx);
    void unsafe_infer_spatial(std::size_t real_idx, std::size_t idx);
    void unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared_p);

    void run_hfa_tiled_inference(std::size_t real_idx, std::size_t idx);

    // MoE inference functions
    void run_moe_infer(std::size_t real_idx, std::size_t idx);
    void run_moe_decoding_inference(std::size_t idx, std::size_t real_idx, const std::vector<size_t>& selected_experts);
    void run_moe_prefill_inference(std::size_t idx,
                                   std::size_t real_idx,
                                   const std::vector<size_t>& selected_experts,
                                   const std::map<size_t, std::vector<size_t>>& token_to_experts,
                                   const std::map<size_t, std::vector<size_t>>& expert_to_tokens);

    void run_moe_prefill_pipeline_inference(std::size_t idx,
                                            std::size_t real_idx,
                                            const std::vector<size_t>& selected_experts,
                                            const std::map<size_t, std::vector<size_t>>& token_to_experts,
                                            const std::map<size_t, std::vector<size_t>>& expert_to_tokens);

    // HFA helper functions
    static void hfa_extract_and_copy_tile(const ov::SoPtr<ov::ITensor>& source_tensor,
                                          const ov::SoPtr<ov::ITensor>& dest_tensor,
                                          uint32_t sequence_dim,
                                          int64_t sequence_offset,
                                          int64_t sequence_length,
                                          const std::string& tensor_name);

    static bool hfa_can_reuse_tensor_zero_copy(const ov::SoPtr<ov::ITensor>& source_tensor,
                                               const ov::SoPtr<ov::ITensor>& dest_tensor,
                                               uint32_t sequence_dim,
                                               int64_t sequence_offset,
                                               int64_t tile_length);

    // MoE helper functions
    void gather_router_scores(const ov::SoPtr<ov::ITensor>& router_source,
                              const ov::SoPtr<ov::ITensor>& router_dest,
                              size_t expert_id,
                              const std::vector<size_t>& token_ids,
                              size_t chunk_start,
                              size_t chunk_size);

    void gather_expert_inputs(const ov::SoPtr<ov::ITensor>& input_source,
                              const ov::SoPtr<ov::ITensor>& input_dest,
                              const std::vector<size_t>& token_ids,
                              size_t chunk_start,
                              size_t chunk_size);

    void scatter_expert_outputs(const ov::SoPtr<ov::ITensor>& expert_output,
                                const std::vector<size_t>& token_ids,
                                size_t chunk_start,
                                size_t chunk_size,
                                size_t expert_id,
                                size_t embed_dim,
                                size_t input_token_count,
                                const std::map<size_t, std::vector<size_t>>& token_to_experts);

    void set_unrolled_router_scores(std::size_t idx, std::size_t real_idx, const std::vector<size_t>& selected_experts);

    void connect_subrequests();
    void recreate_subrequests(std::size_t idx);

    // Helper function to setup pyramid attention infer requests
    void setup_pyramid_infer_requests(std::size_t real_idx, bool is_piped, bool is_recreate);

    // Helper function to setup host flash attention tile infer requests
    void setup_hfa_infer_requests(std::size_t real_idx,
                                  bool is_piped,
                                  bool is_recreate,
                                  bool enable_hfa_optimizations = true);

    FuncMemMgr m_func_mem_mgr;                       // Owns memory
    std::map<LinkFrom, TensorPtr> m_funcall_result;  // Provides a convenient link

    bool is_pipelined(std::size_t idx) const;
    bool m_use_function_pipelining = false;
    struct FuncallPipeline {
        // A "brother" subrequest for a "primary" subrequest. Initialized only
        // for function bodies (replaced_by == idx)
        RqPtr subrequest;

        // Index of the next subrequest in the function call pipeline, if any.
        // Initialized for all funcalls for every function.
        std::optional<std::size_t> next;
    };
    std::vector<std::size_t> m_funcall_heads;

    // This is a sparse vector. It will have the size == number of
    // subgraphs, but with only function call-related elements
    // initialized.
    std::vector<FuncallPipeline> m_funcall_pipeline;

    // Cached check if we do FOLDing and need to update closures in the repeating blocks
    bool m_closure_update_required = false;

    // Cached attention mask for SDPA operations to avoid recomputing
    ov::SoPtr<ov::ITensor> m_cached_attention_mask;

    // HFA runtime context (holds cached masks, pre-allocated buffers, and state buffers)
    std::optional<runtime::host_flash_attention::HFARuntimeContext> m_hfa_runtime_ctx;

    // MoE prefill performance statistics
    struct MoEPrefillStats {
        struct StepStats {
            size_t count = 0;
            double total_ms = 0.0;
            double min_ms = std::numeric_limits<double>::max();
            double max_ms = 0.0;
        };

        StepStats parse_router;      // Parse router output
        StepStats unpack_closure;    // Unpack expert weights
        StepStats set_router_input;  // Set router input tensor
        StepStats expert_inference;  // Expert inference execution
        StepStats dump_tensors;      // Dump input/output to files
        StepStats relayout_output;   // Relayout expert output
        StepStats total_per_expert;  // Total time per expert
        StepStats total_prefill;     // Total prefill time
    };
    MoEPrefillStats m_moe_prefill_stats;

    // MoE decoding performance statistics
    struct MoEDecodingStats {
        struct StepStats {
            size_t count = 0;
            double total_ms = 0.0;
            double min_ms = std::numeric_limits<double>::max();
            double max_ms = 0.0;
        };

        StepStats parse_router;      // Parse router output
        StepStats unpack_closure;    // Unpack batch expert weights
        StepStats set_router_input;  // Set router input tensor
        StepStats expert_inference;  // Expert inference execution
        StepStats total_decoding;    // Total decoding time
    };
    MoEDecodingStats m_moe_decoding_stats;

    // MoE routing maps (reused across inferences to avoid stack allocation)
    std::map<size_t, std::vector<size_t>> m_moe_token_to_experts;
    std::map<size_t, std::vector<size_t>> m_moe_expert_to_tokens;

    // MoE prefill pipeline (defined in moe_prefill_pipeline.hpp)
    std::unique_ptr<MoEPrefillPipeline> m_moe_prefill_pipeline;

    // MoE prefill double-buffering: second infer request for overlapping unpack and inference
    RqPtr m_moe_prefill_second_request;

    // Grant pipeline access to protected members
    friend class MoEPrefillPipeline;
};

}  // namespace npuw
}  // namespace ov
