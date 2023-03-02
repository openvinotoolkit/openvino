// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "dev/converter_utils.hpp"
#include "openvino/runtime/icompiled_model.hpp"

namespace InferenceEngine {

class ICompiledModelWrapper : public ov::ICompiledModel {
public:
    ICompiledModelWrapper(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model);
    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;

    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

    ov::RemoteContext get_context() const override;

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> get_executable_network();

private:
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> m_model;

    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};
}  // namespace InferenceEngine
