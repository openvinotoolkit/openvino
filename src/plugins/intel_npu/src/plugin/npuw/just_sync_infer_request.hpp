// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <vector>

#include "base_sync_infer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "spatial.hpp"

namespace ov {
namespace npuw {

class CompiledModel;
class AsyncInferRequest;

using LinkFrom = std::pair<std::size_t /* Subrequest index */
                           ,
                           std::size_t /* Subrequest output index */
                           >;          // FIXME: This is a third, if not fourth, definitiion of such structure

using TensorPtr = ov::SoPtr<ov::ITensor>;

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

    // Query APIs
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    ////////////////////////////////////
    // implement IBaseInferRequest
    void prepare_for_infer() override;
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t idx) override;
    void run_subrequest_for_success(std::size_t idx, bool& failover) override;
    void subscribe_subrequest(std::size_t idx, Completed cb) override;
    void complete_subrequest(std::size_t idx) override;
    void cancel_subrequest(std::size_t idx) override;
    std::size_t total_subrequests() const override;
    bool supports_async_pipeline() const override;

    void update_subrequest_links(std::size_t idx) override;

    ////////////////////////////////////
    // now own API

    // FIXME: probably this one should go to the base class too
    RqPtr get_real_subrequest(std::size_t idx);

    void bind_global_parameters(std::size_t idx);
    void bind_global_results(std::size_t idx);

    void function_prologue(std::size_t idx);
    void unpack_closure(std::size_t idx, RqPtr request);

    void unsafe_during(std::size_t real_idx, const std::function<void()>& f);
    void unsafe_infer(std::size_t real_idx);
    void unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared_p);

    void connect_subrequests();
    void recreate_subrequests(std::size_t idx);

    TensorPtr allocMem(const ov::element::Type type, const ov::Shape& shape, const std::string& device);
    TensorPtr allocOut(const ov::Output<const ov::Node>& node, const std::string& device);

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

    // This structure tracks how every individual subrequest
    // access the model's top-level (global, public, etc) parameters
    // and results
    struct GlobalIO {
        using map_t = std::map<std::size_t, std::size_t>;
        map_t global_params;   // param idx -> input idx
        map_t global_results;  // result idx -> output idx
    };
    std::vector<GlobalIO> m_subrequests_gio;

    std::unordered_set<void*> m_input_allocated;

    // Represents spatial run-time info
    runtime::spatial::Selector::Ptr m_spatial_selector;

    // Cached check if we do FOLDing and need to update closures in the repeating blocks
    bool m_closure_update_required = false;
};

}  // namespace npuw
}  // namespace ov
