// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "converter_utils.hpp"

#include <ie_blob.h>
#include <ie_common.h>
#include <ie_compound_blob.h>
#include <ie_layouts.h>

#include <fstream>
#include <ie_input_info.hpp>
#include <ie_version.hpp>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/runtime/exception.hpp>
#include <openvino/runtime/remote_context.hpp>
#include <openvino/runtime/tensor.hpp>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp/ie_plugin.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "ie_icore.hpp"
#include "ie_ngraph_utils.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "transformations/utils/utils.hpp"

namespace {

std::string get_legacy_name_from_port(const ov::Output<const ov::Node>& port) {
    ov::Output<ngraph::Node> p(std::const_pointer_cast<ov::Node>(port.get_node_shared_ptr()), port.get_index());
    if (auto node = std::dynamic_pointer_cast<ov::op::v0::Result>(p.get_node_shared_ptr())) {
        p = node->input_value(0);
    }
    return ngraph::op::util::create_ie_output_name(p);
}

void fill_input_info(ov::Output<ov::Node>& input, InferenceEngine::InputInfo::Ptr& input_info) {
    const ov::Output<const ov::Node> const_input(input.get_node(), input.get_index());
    ov::legacy_convert::fill_input_info(const_input, input_info);
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_preproc");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
    it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

void fill_output_info(ov::Output<ov::Node>& input, InferenceEngine::DataPtr& output_info) {
    const ov::Output<const ov::Node> const_input(input.get_node(), input.get_index());
    ov::legacy_convert::fill_output_info(const_input, output_info);
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

InferenceEngine::SizeVector get_dims(const ov::Output<const ov::Node>& port,
                                     const std::function<bool(InferenceEngine::SizeVector& dims)>& callback = {}) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    else {
        if (!callback || !callback(dims)) {
            if (p_shape.rank().is_static()) {
                for (size_t i = 0; i < static_cast<size_t>(p_shape.rank().get_length()); i++) {
                    dims.emplace_back(0);
                }
            }
        }
    }
    return dims;
}

}  // namespace

void ov::legacy_convert::fill_input_info(const ov::Output<const ov::Node>& input,
                                         InferenceEngine::InputInfo::Ptr& input_info) {
    if (!input_info) {
        // Create input info
        auto param_name = input.get_node()->get_friendly_name();
        auto dims = get_dims(input, [&](InferenceEngine::SizeVector& dims) -> bool {
            auto param = std::dynamic_pointer_cast<const ov::op::v0::Parameter>(input.get_node_shared_ptr());
            if (param && param->get_partial_shape().is_static()) {
                dims = param->get_partial_shape().get_shape();
                return true;
            }
            return false;
        });
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(input.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        auto data = std::make_shared<InferenceEngine::Data>(param_name, desc);
        input_info = std::make_shared<InferenceEngine::InputInfo>();
        input_info->setInputData(data);
    }
    auto& rt_info = input.get_rt_info();
    auto it = rt_info.find("ie_legacy_preproc");
    if (it != rt_info.end()) {
        input_info->getPreProcess() = it->second.as<InferenceEngine::PreProcessInfo>();
    }
    it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        input_info->getInputData()->reshape(td.getDims(), td.getLayout());
        input_info->setPrecision(td.getPrecision());
    }
}
void ov::legacy_convert::fill_output_info(const ov::Output<const ov::Node>& output,
                                          InferenceEngine::DataPtr& output_info) {
    if (!output_info) {
        // Create input info
        const auto& res_name = ov::op::util::create_ie_output_name(output);
        auto dims = get_dims(output);
        InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(output.get_element_type()),
                                         dims,
                                         InferenceEngine::TensorDesc::getLayoutByDims(dims));
        output_info = std::make_shared<InferenceEngine::Data>(res_name, desc);
    }
    auto& rt_info = output.get_rt_info();
    auto it = rt_info.find("ie_legacy_td");
    if (it != rt_info.end()) {
        auto td = it->second.as<InferenceEngine::TensorDesc>();
        output_info->reshape(td.getDims(), td.getLayout());
        output_info->setPrecision(td.getPrecision());
    }
}

InferenceEngine::CNNNetwork ov::legacy_convert::convert_model(const std::shared_ptr<const ov::Model>& model,
                                                              bool is_new_api) {
    auto network = InferenceEngine::CNNNetwork(std::shared_ptr<InferenceEngine::ICNNNetwork>(
        new InferenceEngine::details::CNNNetworkNGraphImpl(model->clone(), {}, is_new_api)));
    std::shared_ptr<ov::Model> cloned_model = network.getFunction();
    for (auto&& input : cloned_model->inputs()) {
        auto param_name = input.get_node()->get_friendly_name();

        OPENVINO_ASSERT(network.getInputsInfo().find(param_name) != network.getInputsInfo().end());

        auto input_info = network.getInputsInfo()[param_name];
        ::fill_input_info(input, input_info);
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().find(res_name) != network.getOutputsInfo().end());
        auto output_info = network.getOutputsInfo()[res_name];

        ::fill_output_info(output, output_info);
    }
    return network;
}
std::shared_ptr<const ov::Model> ov::legacy_convert::convert_model(const InferenceEngine::CNNNetwork& network,
                                                                   bool is_new_api) {
    if (is_new_api)
        return network.getFunction();

    auto cloned_model = network.getFunction()->clone();
    for (auto&& input : cloned_model->inputs()) {
        auto param_name = input.get_node()->get_friendly_name();

        OPENVINO_ASSERT(network.getInputsInfo().find(param_name) != network.getInputsInfo().end());

        auto input_info = network.getInputsInfo().at(param_name);
        auto& rt_info = input.get_rt_info();
        rt_info["ie_legacy_preproc"] = input_info->getPreProcess();
        rt_info["ie_legacy_td"] = input_info->getTensorDesc();
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().find(res_name) != network.getOutputsInfo().end());
        auto output_info = network.getOutputsInfo().at(res_name);

        auto& rt_info = output.get_rt_info();
        rt_info["ie_legacy_td"] = output_info->getTensorDesc();
    }
    return cloned_model;
}

namespace ov {

class InferRequestWrapper : public InferenceEngine::IInferRequestInternal {
    ov::Output<const ov::Node> find_port(const std::string& legacy_name) const {
        for (const auto& port : m_request->get_inputs()) {
            if (get_legacy_name_from_port(port) == legacy_name)
                return port;
        }
        for (const auto& port : m_request->get_outputs()) {
            if (get_legacy_name_from_port(port) == legacy_name)
                return port;
        }
        // TODO:
        OPENVINO_ASSERT(false);
    }

public:
    InferRequestWrapper(const std::shared_ptr<ov::IInferRequest>& request) : m_request(request) {}

    std::shared_ptr<ov::IInferRequest> get_infer_request() {
        return m_request;
    }
    void Infer() override {
        m_request->infer();
    }

    void Cancel() override {
        m_request->cancel();
    }

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr& data) override {
        m_request->set_tensor(find_port(name), ov::Tensor{data, {}});
    }

    void SetBlobs(const std::string& name, const std::vector<InferenceEngine::Blob::Ptr>& blobs) override {
        std::vector<ov::Tensor> tensors;
        for (const auto& blob : blobs) {
            tensors.emplace_back(ov::Tensor{blob, {}});
        }
        m_request->set_tensors(find_port(name), tensors);
    }

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override {
        return m_request->get_tensor(find_port(name))._impl;
    }

    InferenceEngine::BatchedBlob::Ptr GetBlobs(const std::string& name) override {
        auto tensors = m_request->get_tensors(find_port(name));
        std::vector<InferenceEngine::Blob::Ptr> blobs;
        for (const auto& tensor : tensors) {
            blobs.emplace_back(tensor._impl);
        }
        return std::make_shared<InferenceEngine::BatchedBlob>(blobs);
    }

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override {
        std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> res;
        auto variables = m_request->query_state();
        for (const auto& var : variables) {
            res.emplace_back(var._impl);
        }
        return res;
    }

    void StartAsync() override {
        m_request->start_async();
    }

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override {
        try {
            // m_request->wait_for(millis_timeout);
        } catch (...) {
            // TODO: FIX
            return InferenceEngine::GENERAL_ERROR;
        }
        return InferenceEngine::OK;
    }

    void SetCallback(Callback callback) override {
        m_request->set_callback(std::move(callback));
    }

private:
    std::shared_ptr<ov::IInferRequest> m_request;
};

class OVIInferRequestWrapper : public ov::IInferRequest {
public:
    OVIInferRequestWrapper(const std::shared_ptr<InferenceEngine::IInferRequestInternal>& request)
        : ov::IInferRequest(
              ov::legacy_convert::convert_compiled_model(request->getPointerToExecutableNetworkInternal())),
          m_request(request) {}

    std::shared_ptr<InferenceEngine::IInferRequestInternal> get_infer_request() {
        return m_request;
    }

    void infer() override {
        m_request->Infer();
    }
    void start_async() override {
        m_request->StartAsync();
    }

    void wait() override {
        try {
            m_request->Wait(ie::InferRequest::RESULT_READY);
        } catch (const ie::InferCancelled& e) {
            throw ov::Cancelled{e.what()};
        } catch (const std::exception& ex) {
            throw Exception(ex.what());
        } catch (...) {
            OPENVINO_UNREACHABLE("Unexpected exception");
        }
    }
    bool wait_for(const std::chrono::milliseconds& timeout) override {
        try {
            return m_request->Wait(timeout.count()) == ie::OK;
        } catch (const ie::InferCancelled& e) {
            throw Cancelled{e.what()};
        } catch (const std::exception& ex) {
            throw Exception(ex.what());
        } catch (...) {
            OPENVINO_UNREACHABLE("Unexpected exception");
        }
    }

    void cancel() override {
        m_request->Cancel();
    }

    std::vector<ov::ProfilingInfo> get_profiling_info() const override {
        auto ieInfos = m_request->GetPerformanceCounts();
        std::vector<ProfilingInfo> infos;
        infos.reserve(ieInfos.size());
        while (!ieInfos.empty()) {
            auto itIeInfo = std::min_element(
                std::begin(ieInfos),
                std::end(ieInfos),
                [](const decltype(ieInfos)::value_type& lhs, const decltype(ieInfos)::value_type& rhs) {
                    return lhs.second.execution_index < rhs.second.execution_index;
                });
            IE_ASSERT(itIeInfo != ieInfos.end());
            auto& ieInfo = itIeInfo->second;
            infos.push_back(ProfilingInfo{});
            auto& info = infos.back();
            switch (ieInfo.status) {
            case ie::InferenceEngineProfileInfo::NOT_RUN:
                info.status = ProfilingInfo::Status::NOT_RUN;
                break;
            case ie::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                info.status = ProfilingInfo::Status::OPTIMIZED_OUT;
                break;
            case ie::InferenceEngineProfileInfo::EXECUTED:
                info.status = ProfilingInfo::Status::EXECUTED;
                break;
            }
            info.real_time = std::chrono::microseconds{ieInfo.realTime_uSec};
            info.cpu_time = std::chrono::microseconds{ieInfo.cpu_uSec};
            info.node_name = itIeInfo->first;
            info.exec_type = std::string{ieInfo.exec_type};
            info.node_type = std::string{ieInfo.layer_type};
            ieInfos.erase(itIeInfo);
        }
        return infos;
    }

    ov::Tensor get_tensor(const ov::Output<const ov::Node>& port) const override {
        const auto& name = get_legacy_name_from_port(port);
        OPENVINO_ASSERT(!m_request->GetBlobs(name),
                        "get_tensor shall not be used together with batched "
                        "set_tensors/set_input_tensors for name '",
                        name,
                        "'");
        auto blob = m_request->GetBlob(name);
        // soVec = {_so, _impl->getPointerToSo()};
        // Tensor tensor = {blob, soVec};
        Tensor tensor = {blob, {}};
        return tensor;
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override {
        m_request->SetBlob(get_legacy_name_from_port(port), tensor._impl);
    }

    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const override {
        auto blobs = m_request->GetBlobs(get_legacy_name_from_port(port));
        std::vector<ov::Tensor> ret;
        for (size_t i = 0; i < blobs->size(); i++) {
            ret.emplace_back(ov::Tensor{blobs->getBlob(i), {}});
        }
        return ret;
    }
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) override {
        std::vector<ie::Blob::Ptr> blobs;
        for (const auto& tensor : tensors) {
            blobs.emplace_back(tensor._impl);
        }
        m_request->SetBlobs(get_legacy_name_from_port(port), blobs);
    }

    std::vector<ov::VariableState> query_state() const override {
        std::vector<ov::VariableState> variable_states;
        std::vector<std::shared_ptr<void>> soVec;
        soVec = {m_request->getPointerToSo()};
        for (auto&& state : m_request->QueryState()) {
            variable_states.emplace_back(ov::VariableState{state, soVec});
        }
        return variable_states;
    }

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_request->SetCallback(std::move(callback));
    }

private:
    std::shared_ptr<InferenceEngine::IInferRequestInternal> m_request;
};

// class OVIPluginWrapper : public ov::IPlugin {
// public:
//     OVIPluginWrapper(const std::shared_ptr<InferenceEngine::IInferencePlugin>& plugin) {
//         auto& ver = plugin->GetVersion();
//         ov::Version version;
//         version.buildNumber = ver.buildNumber;
//         version.description = ver.description;
//         set_version(version);
//         set_name(plugin->GetName());
//         set_core(std::dynamic_pointer_cast<ov::ICore>(plugin->GetCore()));
//     }
//
//     std::shared_ptr<InferenceEngine::IInferencePlugin> get_plugin() {
//         return m_plugin;
//     }
//
//     std::shared_ptr<ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
//                                                   const ov::AnyMap& properties) const override {
//         auto exec_network =
//             m_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()), any_copy(properties));
//         auto compiled_model = ov::legacy_convert::convert_compiled_model(exec_network);
//         return compiled_model;
//     }
//
//     std::shared_ptr<ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
//                                                   const ov::AnyMap& properties,
//                                                   const ov::RemoteContext& context) const override {
//         auto compiled_model = ov::legacy_convert::convert_compiled_model(
//             m_plugin->LoadNetwork(ov::legacy_convert::convert_model(model, is_new_api()),
//                                   any_copy(properties),
//                                   context._impl));
//         return compiled_model;
//     }
//
//     void set_property(const ov::AnyMap& properties) override {
//         m_plugin->SetProperties(properties);
//     }
//
//     ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override {
//         try {
//             return m_plugin->GetConfig(name, arguments);
//         } catch (...) {
//             return m_plugin->GetMetric(name, arguments);
//         }
//     }
//
//     RemoteContext create_context(const ov::AnyMap& remote_properties) const override {
//         return ov::RemoteContext{m_plugin->CreateContext(remote_properties), {nullptr}};
//     }
//
//     RemoteContext get_default_context(const ov::AnyMap& remote_properties) const override {
//         return ov::RemoteContext{m_plugin->GetDefaultContext(remote_properties), {nullptr}};
//     }
//
//     std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const
//     override {
//         return ov::legacy_convert::convert_compiled_model(m_plugin->ImportNetwork(model, any_copy(properties)));
//     }
//
//     std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
//                                                      const ov::RemoteContext& context,
//                                                      const ov::AnyMap& properties) const override {
//         return ov::legacy_convert::convert_compiled_model(
//             m_plugin->ImportNetwork(model, context._impl, any_copy(properties)));
//     }
//
//     const std::shared_ptr<InferenceEngine::ExecutorManager>& get_executor_manager() const override {
//         return m_plugin->executorManager();
//     }
//
//     ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
//                                     const ov::AnyMap& properties) const override {
//         auto res = m_plugin->QueryNetwork(ov::legacy_convert::convert_model(model, is_new_api()),
//         any_copy(properties)); if (res.rc != InferenceEngine::OK) {
//             throw ov::Exception(res.resp.msg);
//         }
//         return res.supportedLayersMap;
//     }
//
//     void add_extension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override {
//         m_plugin->AddExtension(extension);
//     }
//
// private:
//     std::shared_ptr<InferenceEngine::IInferencePlugin> m_plugin;
// };

class IInferencePluginWrapper : public InferenceEngine::IInferencePlugin {
public:
    IInferencePluginWrapper(const std::shared_ptr<ov::IPlugin>& plugin) {
        auto& ver = plugin->get_version();
        InferenceEngine::Version version;
        version.buildNumber = ver.buildNumber;
        version.description = ver.description;
        SetVersion(version);
        _isNewAPI = plugin->is_new_api();
        _executorManager = plugin->get_executor_manager();
    }
    std::string GetName() const noexcept override {
        return m_plugin->get_name();
    }

    void SetName(const std::string& name) noexcept override {
        m_plugin->set_name(name);
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                    ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                    ov::any_copy(config),
                                    ov::RemoteContext{context, {}}));
    }

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const std::string& modelPath,
        const std::map<std::string, std::string>& config) override {
        // FIXME:
        OPENVINO_NOT_IMPLEMENTED;
        // return ov::legacy_convert::convert_compiled_model(
        //     m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
        //                             ov::any_copy(config),
        //                             ov::RemoteContext{context, {}}));
    }

    void AddExtension(const std::shared_ptr<InferenceEngine::IExtension>& extension) override {
        m_plugin->add_extension(extension);
    }

    void SetConfig(const std::map<std::string, std::string>& config) override {
        m_plugin->set_property(ov::any_copy(config));
    }

    void SetProperties(const ov::AnyMap& config) override {
        m_plugin->set_property(config);
    }

    InferenceEngine::Parameter GetConfig(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return m_plugin->get_property(name, options);
    }

    InferenceEngine::Parameter GetMetric(
        const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter>& options) const override {
        return m_plugin->get_property(name, options);
    }

    std::shared_ptr<InferenceEngine::RemoteContext> CreateContext(const InferenceEngine::ParamMap& params) override {
        return m_plugin->create_context(params)._impl;
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(
        const InferenceEngine::ParamMap& params) override {
        return m_plugin->get_default_context(params)._impl;
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        std::ifstream model(modelFileName, std::ios::binary);
        return ov::legacy_convert::convert_compiled_model(m_plugin->import_model(model, ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(m_plugin->import_model(networkModel, ov::any_copy(config)));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            m_plugin->import_model(networkModel, ov::RemoteContext{context, {}}, ov::any_copy(config)));
    }

    void SetCore(std::weak_ptr<InferenceEngine::ICore> core) override {
        return m_plugin->set_core(std::dynamic_pointer_cast<ov::ICore>(core));
    }

    std::shared_ptr<InferenceEngine::ICore> GetCore() const noexcept override {
        auto core = m_plugin->get_core();
        return std::dynamic_pointer_cast<InferenceEngine::ICore>(core);
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto res = m_plugin->query_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                         ov::any_copy(config));
        ie::QueryNetworkResult ret;
        if (!network.getFunction() || res.empty()) {
            ret.rc = InferenceEngine::GENERAL_ERROR;
            return ret;
        }
        ret.supportedLayersMap = res;

        return ret;
    }

    std::shared_ptr<ov::IPlugin> get_plugin() {
        return m_plugin;
    }

private:
    std::shared_ptr<ov::IPlugin> m_plugin;
};

class CompiledModelWrapper : public ov::ICompiledModel {
public:
    CompiledModelWrapper(const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model)
        : ov::ICompiledModel(nullptr, ov::legacy_convert::convert_plugin(model->_plugin)),
          m_model(model) {
        std::vector<ov::Output<const ov::Node>> inputs, outputs;
        for (const auto& input : m_model->getInputs()) {
            inputs.emplace_back(input->output(0));
        }
        for (const auto& output : m_model->getOutputs()) {
            outputs.emplace_back(output->output(0));
        }
        set_inputs(inputs);
        set_outputs(inputs);
    }
    std::shared_ptr<ov::IInferRequest> create_infer_request() const override {
        return ov::legacy_convert::convert_infer_request(m_model->CreateInferRequest());
    }

    void export_model(std::ostream& model) const override {
        m_model->Export(model);
    }

    std::shared_ptr<ov::Model> get_runtime_model() const override {
        return m_model->GetExecGraphInfo();
    }

    void set_property(const ov::AnyMap& properties) override {
        m_model->SetConfig(properties);
    }

    ov::Any get_property(const std::string& name) const override {
        try {
            return m_model->GetMetric(name);
        } catch (ie::Exception&) {
            return m_model->GetConfig(name);
        }
    }

    ov::RemoteContext get_context() const override {
        return {m_model->GetContext(), {}};
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> get_model() {
        return m_model;
    }

private:
    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> m_model;
};

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
        return std::make_shared<InferRequestWrapper>(m_model->create_infer_request());
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
        return std::make_shared<InferRequestWrapper>(m_model->create_infer_request_impl());
    }

    std::shared_ptr<ov::ICompiledModel> get_model() {
        return m_model;
    }

private:
    std::shared_ptr<ov::ICompiledModel> m_model;
};
}  // namespace ov

std::shared_ptr<::InferenceEngine::IInferencePlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::ov::IPlugin>& plugin) {
    if (plugin->old_plugin)
        return plugin->old_plugin;
    // if (auto ie_plugin = std::dynamic_pointer_cast<OVIPluginWrapper>(plugin)) {
    //     return ie_plugin->get_plugin();
    // }
    return std::make_shared<ov::IInferencePluginWrapper>(plugin);
}

std::shared_ptr<::ov::IPlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin) {
    return InferenceEngine::convert_plugin(plugin);
}

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<ov::ICompiledModel>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<CompiledModelWrapper>(model)) {
        return comp_model->get_model();
    }
    return std::make_shared<ov::ExecNetworkWrapper>(model);
}

std::shared_ptr<ov::ICompiledModel> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ExecNetworkWrapper>(model)) {
        return comp_model->get_model();
    }
    return std::make_shared<ov::CompiledModelWrapper>(model);
}

std::shared_ptr<::InferenceEngine::IInferRequestInternal> ov::legacy_convert::convert_infer_request(
    const std::shared_ptr<::ov::IInferRequest>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::OVIInferRequestWrapper>(model)) {
        return comp_model->get_infer_request();
    }
    return std::make_shared<ov::InferRequestWrapper>(model);
}
std::shared_ptr<::ov::IInferRequest> ov::legacy_convert::convert_infer_request(
    const std::shared_ptr<::InferenceEngine::IInferRequestInternal>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::InferRequestWrapper>(model)) {
        return comp_model->get_infer_request();
    }
    return std::make_shared<ov::OVIInferRequestWrapper>(model);
}
