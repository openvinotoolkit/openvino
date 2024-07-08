// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <map>
#include <optional>
#include <vector>

#include "base_sync_infer_request.hpp"

namespace ov {
namespace npuw {

class CompiledModel;
class AsyncInferRequest;

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

    void unsafe_run_this_prep_next(std::size_t idx, bool& next_prepared_p);

    void connect_subrequests();
    void recreate_subrequests(std::size_t idx);

    using LinkFrom = std::pair<std::size_t /* Subrequest index */
                               ,
                               std::size_t /* Subrequest output index */
                               >;          // FIXME: This is a third, if not fourth, definitiion of such structure
    using TensorPtr = ov::SoPtr<ov::ITensor>;
    std::map<LinkFrom, TensorPtr> m_funcall_result;

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
};

}  // namespace npuw
}  // namespace ov
