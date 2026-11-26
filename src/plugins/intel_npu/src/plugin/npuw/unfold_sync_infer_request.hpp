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
#include "intel_npu/common/execution_list.hpp"

namespace ov {
namespace npuw {

class UnfoldInferRequest final : public IBaseInferRequest {
public:
    explicit UnfoldInferRequest(const std::shared_ptr<ov::npuw::CompiledModel>& compiled_model);

    ////////////////////////////////////
    // implement IBaseInferRequest - nether of these are required here
    // this hierarchy needs revew
    void prepare_for_infer() override {}
    bool valid_subrequest(std::size_t idx) const override;
    void start_subrequest(std::size_t) override {}
    void run_subrequest_for_success(std::size_t) override {}
    void subscribe_subrequest(std::size_t, Completed cb) override {}
    void complete_subrequest(std::size_t) override {}
    void cancel_subrequest(std::size_t) override {}
    bool supports_async_pipeline() const override {
        return false;
    }
    void update_subrequest_links(std::size_t) override {}

private:
    void infer() override;

    // PROTOTYPE (see ov::npuw::util::exec_list_enabled_for()): all subgraphs [first, last] recorded into
    // one NPU execution list. When set, infer() runs only through it.
    struct ExecList {
        std::size_t first = 0u;
        std::size_t last = 0u;
        std::size_t executes = 0u;
        std::shared_ptr<::intel_npu::IExecutionList> list;
    };
    void create_execution_list();
    ExecList m_exec_list;
};

}  // namespace npuw
}  // namespace ov
