// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include "ie_remote_context.hpp"
#include "plugin.hpp"

namespace ov {
namespace auto_plugin {
class CompiledModel;
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::auto_plugin::CompiledModel>& compiled_model,
                          const SoAsyncInferRequest& request_to_share_blobs_with);
    ~InferRequest();

    void infer() override;
    std::vector<std::shared_ptr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    const SoAsyncInferRequest& get_shared_request();
    void set_scheduled_request(SoAsyncInferRequest request);
    // Auto-Device impl specific: sets the data (blobs from the device-less requests to the specific device request)
    void set_tensors_to_another_request(const SoAsyncInferRequest& req);
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override;

private:
    SoAsyncInferRequest m_shared_request;
    SoAsyncInferRequest m_scheduled_request;
};
} // namespace auto_plugin
} // namespace ov