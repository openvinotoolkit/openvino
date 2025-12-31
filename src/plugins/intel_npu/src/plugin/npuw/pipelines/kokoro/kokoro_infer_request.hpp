// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "kokoro_compiled_model.hpp"

#include "openvino/core/descriptor/output.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/isync_infer_request.hpp"

namespace ov {
namespace npuw {
    
class KokoroInferRequest: public ov::ISyncInferRequest {
public:
    explicit KokoroInferRequest(const std::shared_ptr<ov::npuw::KokoroCompiledModel>& compiled_model);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;

    void check_tensors() const override {};

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        return {};
    }
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;

protected:
    std::shared_ptr<KokoroCompiledModel> m_kokoro_compiled_model;
    
    std::shared_ptr<ov::IAsyncInferRequest> m_model_a_request;
    std::shared_ptr<ov::IAsyncInferRequest> m_model_b_request;

private:
    void init_tensor(const ov::Output<const ov::Node>& port);
};

}  // namespace npuw
}  // namespace ov