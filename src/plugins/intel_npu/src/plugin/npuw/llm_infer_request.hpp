// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {

class LLMCompiledModel;
class LLMInferRequest final : public ov::ISyncInferRequest {
public:
    explicit LLMInferRequest(const std::shared_ptr<ov::npuw::LLMCompiledModel>& compiled_model);

    void infer() override;

    // I/O APIs - supply default implementations
    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {}
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) {}

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override {}
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override {}

    void check_tensors() const override{};

    virtual std::vector<ov::ProfilingInfo> get_profiling_info() const {}
    virtual std::vector<ov::SoPtr<ov::IVariableState>> query_state() const {}

private:
    std::shared_ptr<ov::IAsyncInferRequest> m_kvcache_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_prefill_request;
};

}  // namespace npuw
}  // namespace ov
