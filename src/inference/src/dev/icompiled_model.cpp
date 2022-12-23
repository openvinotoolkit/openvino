// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/icompiled_model.hpp"

#include <memory>

#include "openvino/core/model.hpp"

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin)
    : m_plugin(plugin) {
    std::shared_ptr<const ov::Model> const_model = model;
    m_inputs = const_model->inputs();
    m_outputs = const_model->outputs();
}

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& exec_network)
    : m_exec_network(exec_network) {
    for (const auto& input : m_exec_network->getInputs()) {
        m_inputs.emplace_back(input->output(0));
    }
    for (const auto& output : m_exec_network->getOutputs()) {
        m_outputs.emplace_back(output->output(0));
    }
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::outputs() const {
    return m_outputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::ICompiledModel::inputs() const {
    return m_inputs;
}
std::shared_ptr<InferenceEngine::IInferRequestInternal> ov::ICompiledModel::create_infer_request() const {
    if (m_exec_network) {
        return m_exec_network->CreateInferRequest();
    }
    std::shared_ptr<InferenceEngine::IInferRequestInternal> asyncRequestImpl =
        create_infer_request_impl(m_inputs, m_outputs);
    // asyncRequestImpl->setPointerToExecutableNetworkInternal(shared_from_this());
    return asyncRequestImpl;
}

void ov::ICompiledModel::export_model(std::ostream& model) const {
    if (m_exec_network)
        m_exec_network->Export(model);
    else
        OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::Model> ov::ICompiledModel::get_runtime_model() const {
    if (m_exec_network)
        return m_exec_network->GetExecGraphInfo();
    else
        OPENVINO_NOT_IMPLEMENTED;
}

void ov::ICompiledModel::set_property(const ov::AnyMap& properties) {
    if (m_exec_network)
        m_exec_network->SetConfig(properties);
    else
        OPENVINO_NOT_IMPLEMENTED;
}

ov::Any ov::ICompiledModel::get_property(const std::string& name) const {
    if (m_exec_network) {
        try {
            return m_exec_network->GetMetric(name);
        } catch (ie::Exception&) {
            return m_exec_network->GetConfig(name);
        }
    } else
        OPENVINO_NOT_IMPLEMENTED;
}

ov::RemoteContext ov::ICompiledModel::get_context() const {
    if (m_exec_network)
        return {m_exec_network->GetContext(), {}};
    else
        OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IInferRequestInternal> ov::ICompiledModel::create_infer_request_impl(
    const std::vector<ov::Output<const ov::Node>>& inputs,
    const std::vector<ov::Output<const ov::Node>>& outputs) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::convert_compiled_model_to_legacy(
    const std::shared_ptr<ov::ICompiledModel>& model) {
    if (model->m_exec_network)
        return model->m_exec_network;
    OPENVINO_NOT_IMPLEMENTED;
}
