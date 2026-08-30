// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "npuw/compiled_model.hpp"

namespace ov::npuw {

// The front-end for the dynamic, stateless PagedAttention model deployed by
// the GenAI continuous-batching pipeline. The model is compiled 1:1 on the PA
// fallback device (CPU; the internal OPENVINO_NPUW_PA_DEVICE env var exists
// for development), NPU*-prefixed properties are held at this level while
// everything else is forwarded to the executing device.
//
// The exposed ports are the inner compiled model's own ports, so the cache
// geometry the pipeline's KVCacheManager reads off them (element types,
// block shapes) is the device-resolved truth with no copying involved.
class PACompiledModel final : public ov::npuw::ICompiledModel {
public:
    PACompiledModel(const std::shared_ptr<ov::Model>& model,
                    const std::shared_ptr<const ov::IPlugin>& plugin,
                    const ov::AnyMap& properties);

    // The wrapper adds no I/O of its own -- it exposes the inner model's ports.
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;

    void export_model(std::ostream& stream) const override;
    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name) const override;

private:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;

    ov::SoPtr<ov::ICompiledModel> m_compiled_model;
};

// 1:1 forwarding request. The ports are shared with the inner request (see
// PACompiledModel), so every call delegates without translation, and the
// request holds no state of its own -- it provides exactly the guarantees of
// using the inner request directly.
class PAInferRequest final : public ov::ISyncInferRequest {
public:
    PAInferRequest(const std::shared_ptr<const ov::ICompiledModel>& compiled_model,
                   ov::SoPtr<ov::IAsyncInferRequest> inner_request);

    void infer() override;

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override;
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override;
    void check_tensors() const override;

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

private:
    ov::SoPtr<ov::IAsyncInferRequest> m_inner_request;
};

}  // namespace ov::npuw
