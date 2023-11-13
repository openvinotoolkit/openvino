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
#include "cpp_interfaces/interface/ie_ivariable_state_internal.hpp"
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
#include "legacy_op_extension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/runtime/exception.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iinfer_request.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/ivariable_state.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/profiling_info.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/runtime/variable_state.hpp"
#include "remote_context_wrapper.hpp"
#include "threading/ie_executor_manager.hpp"
#include "transformations/utils/utils.hpp"

#ifdef PROXY_PLUGIN_ENABLED
#    include "openvino/proxy/infer_request.hpp"
#endif

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
    // Apply parameter substitution here
    // Modify model to unpack string input tensors

    auto cloned = model->clone();

    {
        int model;  // hide this name to avoid accidental access
        //std::cerr << "model->get_parameters().size() = " << cloned->get_parameters().size() << "\n";

        for (size_t i = 0; i < cloned->get_parameters().size(); ++i) {
            //std::cerr << model->get_parameters()[i]->get_friendly_name() << "\n";
            //auto& tensor = model->get_parameters()[i]->output(0).get_tensor();
            //std::cerr << "name before: " << tensor.get_any_name() << "\n";
            //tensor.set_names({tensor.get_any_name() + "/postfix"});
            //std::cerr << "name after: " << tensor.get_any_name() << "\n";
            auto parameter = cloned->get_parameters()[i];
            if(parameter->get_element_type() == element::string) {
                std::cerr << "Detected parameter with name " << parameter->output(0).get_any_name() << " with string type\n";
                // Store shape as a RT attribute, otherwise the validation of next nodes cannot deduce shape from a new parameter
                PartialShape original_shape = parameter->get_partial_shape();
                //parameter->get_rt_info()["original_partial_shape"] = original_shape;  // Not universal in case if the wrapping happens not for Parameter
                parameter->set_element_type(element::u8);
                parameter->set_partial_shape(PartialShape{sizeof(void*)});
                parameter->validate_and_infer_types();

                // Add a new tensor name to recognize this changed parameter in the infer_request
                auto& tensor = parameter->get_output_tensor(0);
                std::string name = tensor.get_any_name();
                name = "__overriden_string_port_prefix__" + name;
                tensor.add_names({name});
                tensor.get_rt_info()["__original_partial_shape"] = original_shape;

                std::cerr << "Patched a parameter of type element::string with new name " << name << "\n";
            }
        }
    }

    //////////////////////////////////////////////


    auto network = InferenceEngine::CNNNetwork(std::shared_ptr<InferenceEngine::ICNNNetwork>(
        new InferenceEngine::details::CNNNetworkNGraphImpl(cloned, {}, is_new_api)));
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

class IVariableStateInternalWrapper : public InferenceEngine::IVariableStateInternal {
    ov::SoPtr<ov::IVariableState> m_state;

public:
    IVariableStateInternalWrapper(const ov::SoPtr<ov::IVariableState>& state)
        : InferenceEngine::IVariableStateInternal(state->get_name()),
          m_state(state) {}

    std::string GetName() const override {
        return m_state->get_name();
    }

    void Reset() override {
        m_state->reset();
    }

    void SetState(const InferenceEngine::Blob::Ptr& newState) override {
        m_state->set_state(ov::make_tensor(newState, true));
    }

    InferenceEngine::Blob::CPtr GetState() const override {
        return tensor_to_blob(m_state->get_state());
    }
};

class IInferencePluginWrapper : public InferenceEngine::IInferencePlugin {
public:
    IInferencePluginWrapper(const ov::SoPtr<ov::IPlugin>& plugin) : m_plugin(plugin) {
        auto& ver = plugin->get_version();
        InferenceEngine::Version version;
        version.buildNumber = ver.buildNumber;
        version.description = ver.description;
        SetVersion(version);
        _isNewAPI = plugin->is_new_api();
        _executorManager = InferenceEngine::create_old_manager(plugin->get_executor_manager());
    }

    virtual ~IInferencePluginWrapper() = default;

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
            {m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                     ov::any_copy(config)),
             m_plugin._so});
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const InferenceEngine::CNNNetwork& network,
        const std::map<std::string, std::string>& config,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context) override {
        return ov::legacy_convert::convert_compiled_model(
            {m_plugin->compile_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                     ov::any_copy(config),
                                     ov::legacy_convert::convert_remote_context(context)),
             m_plugin._so});
    }

    ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> LoadNetwork(
        const std::string& modelPath,
        const std::map<std::string, std::string>& config) override {
        return ov::SoPtr<InferenceEngine::IExecutableNetworkInternal>(
            ov::legacy_convert::convert_compiled_model(
                {m_plugin->compile_model(modelPath, ov::any_copy(config)), m_plugin._so}),
            m_plugin._so);
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
        return ov::legacy_convert::convert_remote_context(m_plugin->create_context(params));
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetDefaultContext(
        const InferenceEngine::ParamMap& params) override {
        return ov::legacy_convert::convert_remote_context(m_plugin->get_default_context(params));
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) override {
        std::ifstream model(modelFileName, std::ios::binary);
        return ov::legacy_convert::convert_compiled_model(
            {m_plugin->import_model(model, ov::any_copy(config)), m_plugin._so});
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            {m_plugin->import_model(networkModel, ov::any_copy(config)), m_plugin._so});
    }

    std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ImportNetwork(
        std::istream& networkModel,
        const std::shared_ptr<InferenceEngine::RemoteContext>& context,
        const std::map<std::string, std::string>& config) override {
        return ov::legacy_convert::convert_compiled_model(
            {m_plugin->import_model(networkModel,
                                    ov::legacy_convert::convert_remote_context(context),
                                    ov::any_copy(config)),
             m_plugin._so});
    }

    void SetCore(std::weak_ptr<InferenceEngine::ICore> core) override {
        return m_plugin->set_core(std::dynamic_pointer_cast<ov::ICore>(core.lock()));
    }

    std::shared_ptr<InferenceEngine::ICore> GetCore() const noexcept override {
        auto core = m_plugin->get_core();
        return std::dynamic_pointer_cast<InferenceEngine::ICore>(core);
    }

    InferenceEngine::QueryNetworkResult QueryNetwork(const InferenceEngine::CNNNetwork& network,
                                                     const std::map<std::string, std::string>& config) const override {
        auto res = m_plugin->query_model(ov::legacy_convert::convert_model(network, m_plugin->is_new_api()),
                                         ov::any_copy(config));
        InferenceEngine::QueryNetworkResult ret;
        if (!network.getFunction() || res.empty()) {
            ret.rc = InferenceEngine::GENERAL_ERROR;
            return ret;
        }
        ret.supportedLayersMap = res;

        return ret;
    }

    ov::SoPtr<ov::IPlugin> get_plugin() {
        return m_plugin;
    }

private:
    ov::SoPtr<ov::IPlugin> m_plugin;
};

}  // namespace ov

std::shared_ptr<::InferenceEngine::IInferencePlugin> ov::legacy_convert::convert_plugin(
    const ov::SoPtr<::ov::IPlugin>& plugin) {
    if (auto wrapper = std::dynamic_pointer_cast<InferenceEngine::IPluginWrapper>(plugin._ptr))
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
    explicit IExecutableNetworkWrapper(const ov::SoPtr<ov::ICompiledModel>& model) : m_model(model) {
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
        _plugin =
            ov::legacy_convert::convert_plugin({std::const_pointer_cast<ov::IPlugin>(m_model->m_plugin), m_model._so});
        _so = model._so;
    }

    std::shared_ptr<InferenceEngine::IInferRequestInternal> CreateInferRequest() override {
        auto infer_request = legacy_convert::convert_infer_request({m_model->create_infer_request(), m_model._so});
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
        // Add legacy supported properties
        if (METRIC_KEY(SUPPORTED_METRICS) == name || METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
            try {
                return m_model->get_property(name);
            } catch (const ov::Exception&) {
                auto props = m_model->get_property(ov::supported_properties.name()).as<std::vector<PropertyName>>();
                std::vector<std::string> legacy_properties;
                for (const auto& prop : props) {
                    if ((METRIC_KEY(SUPPORTED_METRICS) == name && !prop.is_mutable()) ||
                        (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name && prop.is_mutable()))
                        legacy_properties.emplace_back(prop);
                }
                if (METRIC_KEY(SUPPORTED_METRICS) == name) {
                    legacy_properties.emplace_back(METRIC_KEY(SUPPORTED_METRICS));
                    legacy_properties.emplace_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
                }

                return legacy_properties;
            }
        }
        return m_model->get_property(name);
    }

    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override {
        return ov::legacy_convert::convert_remote_context(m_model->get_context());
    }

    ov::SoPtr<ov::ICompiledModel> get_compiled_model() {
        return m_model;
    }

private:
    ov::SoPtr<ov::ICompiledModel> m_model;
};
}  // namespace ov

std::shared_ptr<InferenceEngine::IExecutableNetworkInternal> ov::legacy_convert::convert_compiled_model(
    const ov::SoPtr<ov::ICompiledModel>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<InferenceEngine::ICompiledModelWrapper>(model._ptr)) {
        return comp_model->get_executable_network();
    }
    return std::make_shared<ov::IExecutableNetworkWrapper>(model);
}

ov::SoPtr<ov::ICompiledModel> ov::legacy_convert::convert_compiled_model(
    const std::shared_ptr<InferenceEngine::IExecutableNetworkInternal>& model) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::IExecutableNetworkWrapper>(model)) {
        return comp_model->get_compiled_model();
    }
    return {std::make_shared<InferenceEngine::ICompiledModelWrapper>(model), model->GetPointerToSo()};
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
        OPENVINO_THROW("Failed to find input or output with name: \'", legacy_name, "\'");
    }

public:
    explicit IInferRequestInternalWrapper(const ov::SoPtr<ov::IAsyncInferRequest>& request) : m_request(request) {
        _so = request._so;
    }

    void Infer() override {
        m_request->infer();
    }

    void Cancel() override {
        m_request->cancel();
    }

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override {
        auto res = m_request->get_profiling_info();
        std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> ret;
        for (size_t i = 0; i < res.size(); i++) {
            const auto& info = res[i];
            InferenceEngine::InferenceEngineProfileInfo old_info;
            old_info.cpu_uSec = info.cpu_time.count();
            old_info.execution_index = static_cast<unsigned>(i);
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
        try {
            m_request->set_tensor(find_port(name), ov::make_tensor(data, true));
        } catch (const ov::Exception& ex) {
            const std::string what = ex.what();
            if (what.find("Failed to set tensor") != std::string::npos) {
                IE_THROW(ParameterMismatch) << what;
            }
            IE_THROW(GeneralError) << what;
        }
    }

    void SetBlobs(const std::string& name, const std::vector<InferenceEngine::Blob::Ptr>& blobs) override {
        try {
            std::vector<ov::SoPtr<ov::ITensor>> tensors;
            for (const auto& blob : blobs) {
                tensors.emplace_back(ov::make_tensor(blob, true));
            }
            m_request->set_tensors(find_port(name), tensors);
        } catch (const ov::Exception& ex) {
            IE_THROW(GeneralError) << ex.what();
        }
    }

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override {
        auto port = find_port(name);
        auto& rt_info = port.get_rt_info();
        auto it = rt_info.find("ie_legacy_td");
        InferenceEngine::TensorDesc desc;
        if (it != rt_info.end()) {
            desc = it->second.as<InferenceEngine::TensorDesc>();
        }
        return tensor_to_blob(m_request->get_tensor(port), true, desc);
    }

    InferenceEngine::BatchedBlob::Ptr GetBlobs(const std::string& name) override {
        auto port = find_port(name);
        auto& rt_info = port.get_rt_info();
        auto it = rt_info.find("ie_legacy_td");
        InferenceEngine::TensorDesc desc;
        if (it != rt_info.end()) {
            desc = it->second.as<InferenceEngine::TensorDesc>();
        }
        auto tensors = m_request->get_tensors(port);
        std::vector<InferenceEngine::Blob::Ptr> blobs;
        for (const auto& tensor : tensors) {
            blobs.emplace_back(tensor_to_blob(tensor, true, desc));
        }
        return std::make_shared<InferenceEngine::BatchedBlob>(blobs);
    }

    const InferenceEngine::PreProcessInfo& GetPreProcess(const std::string& name) const override {
#ifdef PROXY_PLUGIN_ENABLED
        if (auto proxy_request = std::dynamic_pointer_cast<ov::proxy::InferRequest>(m_request._ptr)) {
            return ov::legacy_convert::convert_infer_request(proxy_request->get_hardware_request())
                ->GetPreProcess(name);
        }
#endif
        auto port = find_port(name);
        auto& rt_info = port.get_rt_info();
        auto it = rt_info.find("ie_legacy_preproc");
        if (it != rt_info.end()) {
            return it->second.as<InferenceEngine::PreProcessInfo>();
        }
        OPENVINO_THROW("Cannot find PreProcess info.");
    }

    std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> QueryState() override {
        auto res = m_request->query_state();
        std::vector<std::shared_ptr<InferenceEngine::IVariableStateInternal>> ret;
        for (const auto& state : res) {
            ret.emplace_back(std::make_shared<ov::IVariableStateInternalWrapper>(state));
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

    ov::SoPtr<ov::IAsyncInferRequest> get_infer_request() {
        return m_request;
    }

private:
    ov::SoPtr<ov::IAsyncInferRequest> m_request;
};

}  // namespace ov

namespace InferenceEngine {

class IVariableStateWrapper : public ov::IVariableState {
private:
    std::shared_ptr<InferenceEngine::IVariableStateInternal> m_state;

public:
    explicit IVariableStateWrapper(const std::shared_ptr<InferenceEngine::IVariableStateInternal>& state)
        : ov::IVariableState(state->GetName()),
          m_state(state) {}

    void reset() override {
        m_state->Reset();
    }

    void set_state(const ov::SoPtr<ov::ITensor>& state) override {
        m_state->SetState(ov::tensor_to_blob(state));
    }

    ov::SoPtr<ov::ITensor> get_state() const override {
        return ov::make_tensor(std::const_pointer_cast<InferenceEngine::Blob>(m_state->GetState()));
    }
};

class IAsyncInferRequestWrapper : public ov::IAsyncInferRequest {
public:
    IAsyncInferRequestWrapper(const std::shared_ptr<InferenceEngine::IInferRequestInternal>& request,
                              const std::string& plugin_name)
        : ov::IAsyncInferRequest(nullptr, nullptr, nullptr),
          m_request(request),
          m_unwrap_tensor(plugin_name != "AUTO" && plugin_name != "MULTI" && plugin_name != "BATCH" &&
                          plugin_name != "HETERO") {
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
            ov::Cancelled::create(e.what());
        } catch (const std::exception& ex) {
            OPENVINO_THROW(ex.what());
        } catch (...) {
            OPENVINO_THROW("Unexpected exception");
        }
    }
    bool wait_for(const std::chrono::milliseconds& timeout) override {
        try {
            return m_request->Wait(timeout.count()) == InferenceEngine::OK;
        } catch (const InferenceEngine::InferCancelled& e) {
            ov::Cancelled::create(e.what());
        } catch (const std::exception& ex) {
            OPENVINO_THROW(ex.what());
        } catch (...) {
            OPENVINO_THROW("Unexpected exception");
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

    ov::SoPtr<ov::ITensor> get_tensor(const ov::Output<const ov::Node>& port) const override {
        const auto& name = get_legacy_name_from_port(port);
        OPENVINO_ASSERT(!m_request->GetBlobs(name),
                        "get_tensor shall not be used together with batched "
                        "set_tensors/set_input_tensors for name '",
                        name,
                        "'");
        auto blob = m_request->GetBlob(name);
        ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(blob);
        if (!tensor._so)
            tensor._so = m_request->getPointerToSo();
        return tensor;
    }
    void set_tensor(const ov::Output<const ov::Node>& port, const ov::SoPtr<ov::ITensor>& tensor) override {
        m_request->SetBlob(get_legacy_name_from_port(port), ov::tensor_to_blob(tensor, m_unwrap_tensor));
    }

    std::vector<ov::SoPtr<ov::ITensor>> get_tensors(const ov::Output<const ov::Node>& port) const override {
        auto blobs = m_request->GetBlobs(get_legacy_name_from_port(port));
        std::vector<ov::SoPtr<ov::ITensor>> ret;
        if (!blobs)
            return ret;
        for (size_t i = 0; i < blobs->size(); i++) {
            ov::SoPtr<ov::ITensor> tensor = ov::make_tensor(blobs->getBlob(i));
            if (!tensor._so)
                tensor._so = m_request->getPointerToSo();
            ret.emplace_back(tensor);
        }
        return ret;
    }
    void set_tensors(const ov::Output<const ov::Node>& port,
                     const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override {
        std::vector<InferenceEngine::Blob::Ptr> blobs;
        for (const auto& tensor : tensors) {
            blobs.emplace_back(ov::tensor_to_blob(tensor, m_unwrap_tensor));
        }
        m_request->SetBlobs(get_legacy_name_from_port(port), blobs);
    }

    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override {
        std::vector<ov::SoPtr<ov::IVariableState>> variable_states;
        for (auto&& state : m_request->QueryState()) {
            variable_states.push_back(
                {std::make_shared<InferenceEngine::IVariableStateWrapper>(state), m_request->getPointerToSo()});
        }
        return variable_states;
    }

    void set_callback(std::function<void(std::exception_ptr)> callback) override {
        m_request->SetCallback(std::move(callback));
    }

    const std::shared_ptr<const ov::ICompiledModel>& get_compiled_model() const override {
        if (!m_compiled_model) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_compiled_model) {
                if (m_request->getPointerToExecutableNetworkInternal())
                    m_compiled_model =
                        ov::legacy_convert::convert_compiled_model(m_request->getPointerToExecutableNetworkInternal());
            }
        }
        OPENVINO_ASSERT(m_compiled_model);
        return m_compiled_model._ptr;
    }

    const std::vector<ov::Output<const ov::Node>>& get_inputs() const override {
        return get_compiled_model()->inputs();
    }
    const std::vector<ov::Output<const ov::Node>>& get_outputs() const override {
        return get_compiled_model()->outputs();
    }

private:
    std::shared_ptr<InferenceEngine::IInferRequestInternal> m_request;
    mutable ov::SoPtr<const ov::ICompiledModel> m_compiled_model;
    mutable std::mutex m_mutex;
    const bool m_unwrap_tensor;
};

}  // namespace InferenceEngine

std::shared_ptr<::InferenceEngine::IInferRequestInternal> ov::legacy_convert::convert_infer_request(
    const ov::SoPtr<::ov::IAsyncInferRequest>& request) {
    if (auto comp_model = std::dynamic_pointer_cast<InferenceEngine::IAsyncInferRequestWrapper>(request._ptr)) {
        return comp_model->get_infer_request();
    }
    return std::make_shared<ov::IInferRequestInternalWrapper>(request);
}
ov::SoPtr<::ov::IAsyncInferRequest> ov::legacy_convert::convert_infer_request(
    const std::shared_ptr<::InferenceEngine::IInferRequestInternal>& request,
    const std::string& plugin_name) {
    if (auto comp_model = std::dynamic_pointer_cast<ov::IInferRequestInternalWrapper>(request)) {
        return comp_model->get_infer_request();
    }
    return {std::make_shared<InferenceEngine::IAsyncInferRequestWrapper>(request, plugin_name),
            request->getPointerToSo()};
}

namespace InferenceEngine {
const std::shared_ptr<InferenceEngine::RemoteContext>& IRemoteContextWrapper::get_context() {
    return m_context;
}

const std::string& IRemoteContextWrapper::get_device_name() const {
    m_name = m_context->getDeviceName();
    return m_name;
}

const ov::AnyMap& IRemoteContextWrapper::get_property() const {
    m_params = m_context->getParams();
    return m_params;
}

ov::SoPtr<ov::IRemoteTensor> IRemoteContextWrapper::create_tensor(const ov::element::Type& type,
                                                                  const ov::Shape& shape,
                                                                  const ov::AnyMap& params) {
    InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(type),
                                     shape,
                                     InferenceEngine::TensorDesc::getLayoutByDims(shape));
    auto blob = m_context->CreateBlob(desc, params);
    blob->allocate();
    auto tensor = ov::make_tensor(blob);
    return {std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr), tensor._so};
}

ov::SoPtr<ov::ITensor> IRemoteContextWrapper::create_host_tensor(const ov::element::Type type, const ov::Shape& shape) {
    InferenceEngine::TensorDesc desc(InferenceEngine::details::convertPrecision(type),
                                     shape,
                                     InferenceEngine::TensorDesc::getLayoutByDims(shape));
    auto blob = m_context->CreateHostBlob(desc);
    blob->allocate();
    return ov::make_tensor(blob);
}

}  // namespace InferenceEngine

std::shared_ptr<InferenceEngine::RemoteContext> ov::legacy_convert::convert_remote_context(
    const ov::SoPtr<ov::IRemoteContext>& context) {
    if (auto ctx = std::dynamic_pointer_cast<InferenceEngine::IRemoteContextWrapper>(context._ptr)) {
        return ctx->get_context();
    }
    return std::make_shared<ov::RemoteContextWrapper>(context);
}

ov::SoPtr<ov::IRemoteContext> ov::legacy_convert::convert_remote_context(
    const std::shared_ptr<InferenceEngine::RemoteContext>& context) {
    if (auto ctx = std::dynamic_pointer_cast<ov::RemoteContextWrapper>(context)) {
        return ctx->get_context();
    }
    return {std::make_shared<InferenceEngine::IRemoteContextWrapper>(context)};
}

namespace ov {

/*
 * @brief Wrapper for old IE extensions to new API
 */
class ExtensionWrapper : public ov::LegacyOpExtension {
public:
    ExtensionWrapper(const InferenceEngine::IExtensionPtr& ext, const std::string& opset, const std::string& name)
        : m_ext(ext),
          m_opset_name(opset),
          m_type(name),
          m_ext_type(m_type.c_str(), m_opset_name.c_str()) {}
    ~ExtensionWrapper() override = default;

    const ov::DiscreteTypeInfo& get_type_info() const override {
        return m_ext_type;
    }

    ngraph::OutputVector create(const ngraph::OutputVector& inputs, ngraph::AttributeVisitor& visitor) const override {
        std::shared_ptr<ngraph::Node> node(m_ext->getOpSets().at(m_opset_name).create_insensitive(m_ext_type.name));

        node->set_arguments(inputs);
        if (node->visit_attributes(visitor)) {
            node->constructor_validate_and_infer_types();
        }
        return node->outputs();
    }

    std::vector<ov::Extension::Ptr> get_attached_extensions() const override {
        return {};
    }

    const InferenceEngine::IExtensionPtr& get_extension() const {
        return m_ext;
    }

private:
    InferenceEngine::IExtensionPtr m_ext;
    std::string m_opset_name;
    std::string m_type;
    ov::DiscreteTypeInfo m_ext_type;
};

}  // namespace ov

std::vector<ov::Extension::Ptr> ov::legacy_convert::convert_extension(
    const std::vector<InferenceEngine::IExtensionPtr>& exts) {
    std::vector<ov::Extension::Ptr> extensions;
    for (const auto& ext : exts) {
        for (const auto& item : ext->getOpSets()) {
            for (const auto& type_info : item.second.get_types_info()) {
                extensions.emplace_back(std::make_shared<ov::ExtensionWrapper>(ext, item.first, type_info.name));
            }
        }
    }
    return extensions;
}

std::vector<InferenceEngine::IExtensionPtr> ov::legacy_convert::convert_extension(
    const std::vector<ov::Extension::Ptr>& exts) {
    std::vector<InferenceEngine::IExtensionPtr> extensions;
    std::unordered_set<InferenceEngine::IExtensionPtr> existed_extensions;
    for (const auto& ext : exts) {
        if (const auto& wrapper = std::dynamic_pointer_cast<ov::ExtensionWrapper>(ext)) {
            if (!existed_extensions.count(wrapper->get_extension())) {
                extensions.emplace_back(wrapper->get_extension());
                existed_extensions.insert(wrapper->get_extension());
            }
        }
    }
    return extensions;
}
