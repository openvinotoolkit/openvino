// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "attn/attn_subgraph.hpp"
#include "base_sync_infer_request.hpp"
#include "host_flash_attention.hpp"
#include "moe/moe_executor.hpp"
#include "moe/moe_infer_utils.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "spatial.hpp"
#include "v1/subgraph_pipeline.hpp"

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

class JustInferRequest final : public IBaseInferRequest, public ISubrequestAccessor {
public:
    explicit JustInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    ////////////////////////////////////
    // Implement ISubrequestAccessor interface
    ov::SoPtr<ov::IAsyncInferRequest> get_subrequest(size_t idx) override;
    const void* get_submodel_desc(size_t idx) override;
    TensorPtr allocate_mem(const ov::element::Type& type, const ov::Shape& shape, const std::string& device) override;
    bool is_gather_closure(size_t idx, size_t cidx) override;
    bool unpack_required(size_t idx, size_t cidx) override;
    bool needs_copy_closure(size_t idx, size_t cidx) override;
    std::string subgraph_device(size_t idx) override;
    void set_active_subrequest(size_t idx, ov::SoPtr<ov::IAsyncInferRequest> request);
    ov::SoPtr<ov::IAsyncInferRequest> get_pipeline_subrequest(size_t idx) const;
    void set_pipeline_subrequest(size_t idx, ov::SoPtr<ov::IAsyncInferRequest> request);
    bool is_subrequest_pipelined(size_t idx) const;
    std::size_t history_size() const;
    bool subgraph_needs_copy(std::size_t idx) const;
    bool attention_no_copy() const;
    const ov::SoPtr<ov::ICompiledModel>& compiled_submodel(size_t idx) const;
    const ov::npuw::v1::subgraphs::CompiledPipeline& subgraph_pipeline(size_t idx) const;
    std::size_t subgraph_param_base(size_t idx) const;

protected:
    ////////////////////////////////////
    // implement IBaseInferRequest
    void prepare_for_infer() override;
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t idx) override;
    void run_subrequest_for_success(std::size_t idx) override;
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
    bool bind_behavior_input(std::size_t idx,
                             std::size_t real_idx,
                             std::size_t input_idx,
                             const ov::SoPtr<ov::ITensor>& tensor,
                             RqPtr request) override;

    void function_prologue(std::size_t idx);

    void unsafe_during(std::size_t real_idx, std::size_t idx, const std::function<void()>& f);
    void unsafe_infer(std::size_t real_idx, std::size_t idx);
    void unsafe_infer_spatial(std::size_t real_idx, std::size_t idx);
    void unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared_p);

    void legacy_infer(std::size_t real_idx, std::size_t idx);
    ov::npuw::v1::subgraphs::InferContext make_behavior_context(std::size_t real_idx, std::size_t idx);
    const ov::npuw::v1::subgraphs::RuntimeBehaviorSpec* get_runtime_behavior_spec(std::size_t idx) const;
    ov::npuw::v1::subgraphs::ISubgraphBehavior* get_subgraph_behavior(std::size_t idx) const;
    bool behavior_handles_function_prologue(std::size_t idx) const;

protected:
    void connect_subrequests();
    void initialize_subgraph_behaviors();

    // Helper function to initialize/reinitialize MoE executor
    void initialize_moe_executor();

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

    // MoE executor (encapsulates MoE inference logic and profiling)
    std::unique_ptr<ov::npuw::moe::MoEExecutor> m_moe_executor;

    std::vector<ov::npuw::v1::subgraphs::ISubgraphBehavior::Ptr> m_subgraph_behaviors;
    std::vector<ov::npuw::v1::subgraphs::Context> m_subgraph_runtime_states;
};

}  // namespace npuw
}  // namespace ov
