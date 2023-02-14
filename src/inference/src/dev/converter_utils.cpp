// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "converter_utils.hpp"

#include <fstream>
#include <memory>
#include <mutex>

#include "any_copy.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "icompiled_model_wrapper.hpp"
#include "ie_blob.h"
#include "ie_common.h"
#include "ie_compound_blob.h"
#include "ie_icore.hpp"
#include "ie_input_info.hpp"
#include "ie_layouts.h"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_version.hpp"
#include "iplugin_wrapper.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/variable_state.hpp"
#include "so_ptr.hpp"
#include "transformations/utils/utils.hpp"

namespace {

std::string get_legacy_name_from_port(const ov::Output<const ov::Node>& port) {
    ov::Output<ngraph::Node> p(std::const_pointer_cast<ov::Node>(port.get_node_shared_ptr()), port.get_index());
    if (auto node = std::dynamic_pointer_cast<ov::op::v0::Result>(p.get_node_shared_ptr())) {
        p = node->input_value(0);
    }
    return ov::op::util::create_ie_output_name(p);
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

InferenceEngine::SizeVector get_dims(const ov::Output<const ov::Node>& port) {
    InferenceEngine::SizeVector dims = {};
    const auto& p_shape = port.get_partial_shape();
    if (p_shape.is_static())
        dims = p_shape.get_shape();
    return dims;
}

}  // namespace

void ov::legacy_convert::fill_input_info(const ov::Output<const ov::Node>& input,
                                         InferenceEngine::InputInfo::Ptr& input_info) {
    if (!input_info) {
        // Create input info
        auto param_name = input.get_node()->get_friendly_name();
        auto dims = get_dims(input);
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

        OPENVINO_ASSERT(network.getInputsInfo().count(param_name));

        auto input_info = network.getInputsInfo()[param_name];
        ::fill_input_info(input, input_info);
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().count(res_name));
        auto output_info = network.getOutputsInfo()[res_name];

        ::fill_output_info(output, output_info);
    }
    return network;
}
std::shared_ptr<const ov::Model> ov::legacy_convert::convert_model(const InferenceEngine::CNNNetwork& network,
                                                                   bool is_new_api) {
    OPENVINO_ASSERT(network.getFunction(),
                    "CNNNetwork can be converted to OpenVINO Model only in case if it contains ngraph::Function");
    if (is_new_api)
        return network.getFunction();

    auto cloned_model = network.getFunction()->clone();
    for (auto&& input : cloned_model->inputs()) {
        auto param_name = input.get_node()->get_friendly_name();

        OPENVINO_ASSERT(network.getInputsInfo().count(param_name));

        auto input_info = network.getInputsInfo().at(param_name);
        auto& rt_info = input.get_rt_info();
        rt_info["ie_legacy_preproc"] = input_info->getPreProcess();
        rt_info["ie_legacy_td"] = input_info->getTensorDesc();
    }
    for (auto&& result : cloned_model->get_results()) {
        auto output = result->input_value(0);
        const auto& res_name = ov::op::util::create_ie_output_name(output);

        OPENVINO_ASSERT(network.getOutputsInfo().count(res_name));
        auto output_info = network.getOutputsInfo().at(res_name);

        auto& rt_info = output.get_rt_info();
        rt_info["ie_legacy_td"] = output_info->getTensorDesc();
    }
    if (!cloned_model->has_rt_info("version")) {
        cloned_model->set_rt_info(int64_t(10), "version");
    }
    return cloned_model;
}

namespace ov {

class IInferencePluginWrapper : public InferenceEngine::IInferencePlugin {
public:
    IInferencePluginWrapper(const std::shared_ptr<ov::IPlugin>& plugin) : m_plugin(plugin) {
        auto& ver = plugin->get_version();
        InferenceEngine::Version version;
        version.buildNumber = ver.buildNumber;
        version.description = ver.description;
        SetVersion(version);
        _isNewAPI = plugin->is_new_api();
        _executorManager = plugin->get_executor_manager();
    }
    std::string GetName() const noexcept override {
        return m_plugin->get_device_name();
    }

    void SetName(const std::string& name) noexcept override {
        m_plugin->set_device_name(name);
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
        return ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(
            ov::legacy_convert::convert_compiled_model(m_plugin->compile_model(modelPath, ov::any_copy(config))),
            {});
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

}  // namespace ov

std::shared_ptr<::InferenceEngine::IInferencePlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::ov::IPlugin>& plugin) {
    if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin))
        return wrapper->get_plugin();
    return std::make_shared<ov::IInferencePluginWrapper>(plugin);
}

std::shared_ptr<::ov::IPlugin> ov::legacy_convert::convert_plugin(
    const std::shared_ptr<::InferenceEngine::IInferencePlugin>& plugin) {
    std::shared_ptr<::ov::IPlugin> ov_plugin(new ::InferenceEngine::IPluginWrapper(plugin));
    return ov_plugin;
}

namespace ov {

class IExecutableNetworkWrapper : public InferenceEngine::IExecutableNetworkInternal {
public:
    explicit IExecutableNetworkWrapper(const std::shared_ptr<ov::ICompiledModel>& model) : m_model(model) {
        for (const auto& input : m_model->inputs()) {
            InferenceEngine::InputInfo::Ptr input_info;
            ov::legacy_convert::fill_input_info(input, input_info);
            _networkInputs[input_info->name()] = input_info;
            _parameters.emplace_back(input.get_node_shared_ptr());
        }
        for (const auto& output : m_model->outputs()) {
            auto out = output.get_node()->input_value(0);
            InferenceEngine::DataPtr output_info;
            ov::legacy_convert::fill_output_info(ov::Output<const ov::Node>(out.get_node(), out.get_index()),
                                                 output_info);
            _networkOutputs[output_info->getName()] = output_info;
            _results.emplace_back(output.get_node_shared_ptr());
        }
        _plugin = ov::legacy_convert::convert_plugin(std::const_pointer_cast<ov::IPlugin>(m_model->m_plugin));
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequest() override {
        auto infer_request = legacy_convert::convert_infer_request(m_model->create_infer_request());
        infer_request->setPointerToExecutableNetworkInternal(shared_from_this());
        return infer_request;
    }

    void Export(std::ostream& model) override {
        m_model->export_model(model);
    }

    void Export(const std::string& modelFileName) override {
        std::ofstream ostream(modelFileName, std::ios::out | std::ios::binary);
        Export(ostream);
    }

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override {
        return m_model->get_runtime_model()->clone();
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

    std::shared_ptr<ov::ICompiledModel> get_compiled_model() {
        return m_model;
    }

private:
    std::shared_ptr<ov::ICompiledModel> m_model;
};
}  // namespace ov

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<ov::ICompiledModel>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<InferenceEngine::ICompiledModelWrapper>(model)) {
        return comp_model->get_executable_network();
    }
    return std::make_shared<ov::IExecutableNetworkWrapper>(model);
}

std::shared_ptr<ov::ICompiledModel> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::IExecutableNetworkWrapper>(model)) {
        return comp_model->get_compiled_model();
    }
    return std::make_shared<InferenceEngine::ICompiledModelWrapper>(model);
}

namespace ov {

class IInferRequestInternalWrapper : public InferenceEngine::IInferRequestInternal {
    ov::Output<const ov::Node> find_port(const std::string& legacy_name) const {
        for (const auto& port : m_request->get_inputs()) {
            if (get_legacy_name_from_port(port) == legacy_name)
                return port;
        }
        for (const auto& port : m_request->get_outputs()) {
            if (get_legacy_name_from_port(port) == legacy_name)
                return port;
        }
        OPENVINO_ASSERT(false, "Cannot find port with name: ", legacy_name);
    }

public:
    explicit IInferRequestInternalWrapper(const std::shared_ptr<ov::IAsyncInferRequest>& request)
        : m_request(request) {}

    void Infer() override {
        m_request->infer();
    }

    void Cancel() override {
        m_request->cancel();
    }

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        auto res = m_request->get_profiling_info();
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> ret;
        for (const auto& info : res) {
            InferenceEngine::InferenceEngineProfileInfo old_info;
            old_info.cpu_uSec = info.cpu_time.count();
            old_info.realTime_uSec = info.real_time.count();
            strncpy(old_info.exec_type, info.exec_type.c_str(), sizeof(old_info.exec_type));
            old_info.exec_type[sizeof(old_info.exec_type) - 1] = 0;
            strncpy(old_info.layer_type, info.node_type.c_str(), sizeof(old_info.layer_type));
            old_info.layer_type[sizeof(old_info.layer_type) - 1] = 0;
            switch (info.status) {
            case ov::ProfilingInfo::Status::EXECUTED:
                old_info.status = InferenceEngine::InferenceEngineProfileInfo::EXECUTED;
                break;
            case ov::ProfilingInfo::Status::NOT_RUN:
                old_info.status = InferenceEngine::InferenceEngineProfileInfo::NOT_RUN;
                break;
            case ov::ProfilingInfo::Status::OPTIMIZED_OUT:
                old_info.status = InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT;
                break;
            }
            ret[info.node_name] = old_info;
        }
        return ret;
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

    void SetBlob(const std::string& name,
                 const InferenceEngine::Blob::Ptr& data,
                 const InferenceEngine::PreProcessInfo& info) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    const InferenceEngine::PreProcessInfo& GetPreProcess(const std::string& name) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    void SetBatch(int batch) override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override {
        auto res = m_request->query_state();
        std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> ret;
        for (const auto& state : res) {
            ret.emplace_back(state._impl);
        }
        return ret;
    }

    void StartAsync() override {
        m_request->start_async();
    }

    InferenceEngine::StatusCode Wait(int64_t millis_timeout) override {
        if (millis_timeout == InferenceEngine::IInferRequest::RESULT_READY) {
            m_request->wait();
        } else {
            std::chrono::milliseconds timeout(millis_timeout);
            bool res = m_request->wait_for(timeout);
            if (!res)
                return InferenceEngine::StatusCode::RESULT_NOT_READY;
        }
        return InferenceEngine::StatusCode::OK;
    }

    void SetCallback(std::function<void(std::exception_ptr)> callback) override {
        m_request->set_callback(std::move(callback));
    }

    std::shared_ptr<ov::IAsyncInferRequest> get_infer_request() {
        return m_request;
    }

private:
    std::shared_ptr<ov::IAsyncInferRequest> m_request;
};

}  // namespace ov

namespace InferenceEngine {

class IAsyncInferRequestWrapper : public ov::IAsyncInferRequest {
public:
    IAsyncInferRequestWrapper(const std::shared_ptr<InferenceEngine::IInferRequestInternal>& request)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request) {
        if (m_request->getPointerToExecutableNetworkInternal())
            m_compiled_model =
                ov::legacy_convert::convert_compiled_model(m_request->getPointerToExecutableNetworkInternal());
    }
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
            m_request->Wait(InferenceEngine::InferRequest::RESULT_READY);
        } catch (const ov::Cancelled&) {
            throw;
        } catch (const InferenceEngine::InferCancelled& e) {
            throw ov::Cancelled{e.what()};
        } catch (const std::exception& ex) {
            throw ov::Exception(ex.what());
        } catch (...) {
            OPENVINO_UNREACHABLE("Unexpected exception");
        }
    }
    bool wait_for(const std::chrono::milliseconds& timeout) override {
        try {
            return m_request->Wait(timeout.count()) == InferenceEngine::OK;
        } catch (const InferenceEngine::InferCancelled& e) {
            throw ov::Cancelled{e.what()};
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
        std::vector<ov::ProfilingInfo> infos;
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
            infos.push_back(ov::ProfilingInfo{});
            auto& info = infos.back();
            switch (ieInfo.status) {
            case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
                info.status = ov::ProfilingInfo::Status::NOT_RUN;
                break;
            case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
                info.status = ov::ProfilingInfo::Status::OPTIMIZED_OUT;
                break;
            case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
                info.status = ov::ProfilingInfo::Status::EXECUTED;
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
        ov::Tensor tensor = {blob, {m_request->getPointerToSo()}};
        return tensor;
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::Tensor& tensor) override {
        m_request->SetBlob(get_legacy_name_from_port(port), tensor._impl);
    }

    std::vector<ov::Tensor> get_tensors(const ov::Output<const ov::Node>& port) const override {
        auto blobs = m_request->GetBlobs(get_legacy_name_from_port(port));
        std::vector<ov::Tensor> ret;
        if (!blobs)
            return ret;
        for (size_t i = 0; i < blobs->size(); i++) {
            ret.emplace_back(ov::Tensor{blobs->getBlob(i), {m_request->getPointerToSo()}});
        }
        return ret;
    }
    void set_tensors(const ov::Output<const ov::Node>& port, const std::vector<ov::Tensor>& tensors) override {
        std::vector<InferenceEngine::Blob::Ptr> blobs;
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

    const std::shared_ptr<ov::ICompiledModel>& get_compiled_model() const override {
        if (!m_compiled_model) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_compiled_model) {
                if (m_request->getPointerToExecutableNetworkInternal())
                    m_compiled_model =
                        ov::legacy_convert::convert_compiled_model(m_request->getPointerToExecutableNetworkInternal());
            }
        }
        OPENVINO_ASSERT(m_compiled_model);
        return m_compiled_model;
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return get_compiled_model()->inputs();
    }
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return get_compiled_model()->outputs();
    }

private:
    std::shared_ptr<InferenceEngine::IInferRequestInternal> m_request;
    mutable std::shared_ptr<ov::ICompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
};

}  // namespace InferenceEngine

std::shared_ptr<::InferenceEngine::IInferRequestInternal> ov::legacy_convert::convert_infer_request(
    const std::shared_ptr<::ov::IAsyncInferRequest>& request) {
    if (auto comp_model = std::dynamic_pointer_cast<InferenceEngine::IAsyncInferRequestWrapper>(request)) {
        return comp_model->get_infer_request();
    }
    return std::make_shared<ov::IInferRequestInternalWrapper>(request);
}
std::shared_ptr<::ov::IAsyncInferRequest> ov::legacy_convert::convert_infer_request(
    const std::shared_ptr<::InferenceEngine::IInferRequestInternal>& request) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::IInferRequestInternalWrapper>(request)) {
        return comp_model->get_infer_request();
    }
    return std::make_shared<InferenceEngine::IAsyncInferRequestWrapper>(request);
}
