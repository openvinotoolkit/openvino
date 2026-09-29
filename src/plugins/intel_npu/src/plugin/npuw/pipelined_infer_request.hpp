// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "base_sync_infer_request.hpp"

namespace ov {
namespace npuw {

class PipelinedInferRequest final : public IBaseInferRequest {
public:
    explicit PipelinedInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;

protected:
    void prepare_for_infer() override;
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t idx) override;
    void run_subrequest_for_success(std::size_t idx) override;
    void subscribe_subrequest(std::size_t idx, Completed cb) override;
    void complete_subrequest(std::size_t idx) override;
    void cancel_subrequest(std::size_t idx) override;
    bool supports_async_pipeline() const override;
    void update_subrequest_links(std::size_t idx) override;

    void unpack_closure(std::size_t idx, RqPtr request) override;
    void bind_global_params(std::size_t idx, RqPtr request) override;
    void bind_global_results(std::size_t idx, RqPtr request) override;
    void handle_quant_host_gather(std::size_t idx, RqPtr request) override;
    void alloc_quant_gather() override;

private:
    void initialize_shared_inputs();
    void initialize_outputs();
    void initialize_hfa_branch_selection();
    std::optional<ov::Output<const ov::Node>> get_pipeline_parameter_port(std::size_t idx,
                                                                          std::size_t closure_param_id) const;
    std::optional<ov::Output<const ov::Node>> get_pipeline_port(const std::string& port_name,
                                                                std::size_t usage_index) const;
    void update_hfa_branch_selection();

    RqPtr m_pipeline_request;
};

}  // namespace npuw
}  // namespace ov