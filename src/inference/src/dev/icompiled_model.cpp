// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/icompiled_model.hpp"

#include <cpp/ie_executable_network.hpp>
#include <ie_remote_context.hpp>
#include <memory>
#include <openvino/runtime/remote_context.hpp>

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "dev/converter_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace ov {

class ExecNetworkWrapper : public InferenceEngine::IExecutableNetworkInternal {
public:
    ExecNetworkWrapper(const std::shared_ptr<ov::ICompiledModel>& model) : m_model(model) {
        for (const auto& input : m_model->inputs()) {
            InferenceEngine::InputInfo::Ptr input_info;
            ov::legacy_convert::fill_input_info(input, input_info);
            _networkInputs[input_info->name()] = input_info;
            _parameters.emplace_back(input.get_node_shared_ptr());
        }
        for (const auto& output : m_model->outputs()) {
            InferenceEngine::DataPtr output_info;
            ov::legacy_convert::fill_output_info(output, output_info);
            _networkOutputs[output_info->getName()] = output_info;
            _results.emplace_back(output.get_node_shared_ptr());
        }
        _plugin = ov::legacy_convert::convert_plugin(std::const_pointer_cast<ov::IPlugin>(m_model->m_plugin));
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequest() override {
        return m_model->create_infer_request();
    }

    void Export(std::ostream& model) override {
        m_model->export_model(model);
    }

    void Export(const std::string& modelFileName) override {
        std::ofstream ostream(modelFileName, std::ios::out | std::ios::binary);
        Export(ostream);
    }

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override {
        return m_model->get_runtime_model();
    }

    void SetConfig(const std::map<std::string, InferenceEngine::Parameter>& config) override {
        m_model->set_property(config);
    }

    InferenceEngine::Parameter GetConfig(const std::string& name) const override {
        return m_model->get_property(name);
    }

    InferenceEngine::Parameter GetMetric(const std::string& name) const override {
        return m_model->get_property(name);
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override {
        return m_model->get_context()._impl;
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequestImpl(
        InferenceEngine::InputsDataMap networkInputs,
        InferenceEngine::OutputsDataMap networkOutputs) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequestImpl(
        const std::vector<std::shared_ptr<const ov::Node>>& inputs,
        const std::vector<std::shared_ptr<const ov::Node>>& outputs) override {
        // TODO: chect that inputs, outputs == m_inputs, m_outputs
        // std::vector<ov::Output<const ov::Node>> model_inputs, model_outputs;
        // for (const auto& input : inputs) {
        //     model_inputs.emplace_back(ov::Output<const ov::Node>{input, 0});
        // }
        // for (const auto& output : outputs) {
        //     ov::Output<ov::Node> in_value = output->input_value(0);
        //     model_outputs.emplace_back(ov::Output<const ov::Node>{in_value.get_node(), 0});
        // }
        return m_model->create_infer_request_impl();
    }

private:
    std::shared_ptr<ov::ICompiledModel> m_model;
};

}  // namespace ov

ov::ICompiledModel::ICompiledModel(const std::shared_ptr<ov::Model>& model,
                                   const std::shared_ptr<const ov::IPlugin>& plugin,
                                   const InferenceEngine::ITaskExecutor::Ptr& task_executor,
                                   const InferenceEngine::ITaskExecutor::Ptr& callback_executor)
    : m_plugin(plugin),
      m_task_executor(task_executor),
      m_callback_executor(callback_executor) {
    std::shared_ptr<const ov::Model> const_model = model;
    // Add pre-processing
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
    if (m_task_executor && m_callback_executor) {
        return create_async_infer_request_from_sync();
    }
    return create_infer_request_impl();
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

std::shared_ptr<InferenceEngine::IInferRequestInternal> ov::ICompiledModel::create_infer_request_impl() const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::convert_compiled_model_to_legacy(
    const std::shared_ptr<ov::ICompiledModel>& model) {
    if (model->m_exec_network)
        return model->m_exec_network;
    return std::make_shared<ExecNetworkWrapper>(model);
}
