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
    void run_subrequest_for_success(std::size_t, bool&) override {}
    void subscribe_subrequest(std::size_t, Completed cb) override {}
    void complete_subrequest(std::size_t) override {}
    void cancel_subrequest(std::size_t) override {}
    bool supports_async_pipeline() const override {
        return false;
    }
    void update_subrequest_links(std::size_t) override {}

private:
    void infer() override;
};

}  // namespace npuw
}  // namespace ov
