// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_core.hpp"

#include <sys/stat.h>

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "any_copy.hpp"
#include "cache_guard.hpp"
#include "check_network_batchable.hpp"
#include "cnn_network_ngraph_impl.hpp"
#include "compilation_context.hpp"
#include "cpp/ie_cnn_network.h"
#include "cpp_interfaces/interface/ie_iexecutable_network_internal.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "dev/converter_utils.hpp"
#include "dev/core_impl.hpp"
#include "file_utils.h"
#include "ie_cache_manager.hpp"
#include "ie_icore.hpp"
#include "ie_network_reader.hpp"
#include "ie_ngraph_utils.hpp"
#include "ie_plugin_config.hpp"
#include "ie_remote_context.hpp"
#include "itt.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/so_extension.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/threading/executor_manager.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/shared_object.hpp"
#include "xml_parse_utils.h"

using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine;
using namespace std::placeholders;

namespace {

std::tuple<bool, std::string> CheckStatic(const InferenceEngine::CNNNetwork& network) {
    bool res = true;
    std::stringstream errMsg;
    auto model = network.getFunction();
    if (model) {
        for (const auto& input : model->inputs()) {
            if (input.get_partial_shape().is_dynamic()) {
                errMsg << "{ input:'";
                for (const auto& name : input.get_names()) {
                    errMsg << name << ",";
                }
                if (auto node = input.get_node_shared_ptr()) {
                    errMsg << node->get_friendly_name();
                }
                errMsg << "', shape=" << input.get_partial_shape() << "} ";
                res = false;
            }
        }
    }
    return {res, errMsg.str()};
}
}  // namespace

namespace InferenceEngine {

class Core::Impl : public ov::CoreImpl {
public:
    Impl() : ov::CoreImpl(false) {}
};

Core::Core(const std::string& xmlConfigFile) {
    _impl = std::make_shared<Impl>();

    try {
        std::string xmlConfigFile_ = ov::find_plugins_xml(xmlConfigFile);
        if (!xmlConfigFile_.empty())
            // If XML is default, load default plugins by absolute paths
            _impl->register_plugins_in_registry(xmlConfigFile_, xmlConfigFile.empty());
        // Load plugins from pre-compiled list
        _impl->register_compile_time_plugins();
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

std::map<std::string, Version> Core::GetVersions(const std::string& deviceName) const {
    return _impl->GetVersions(deviceName);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

CNNNetwork Core::ReadNetwork(const std::wstring& modelPath, const std::wstring& binPath) const {
    try {
        return ReadNetwork(ov::util::wstring_to_string(modelPath), ov::util::wstring_to_string(binPath));
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

#endif

CNNNetwork Core::ReadNetwork(const std::string& modelPath, const std::string& binPath) const {
    try {
        return _impl->ReadNetwork(modelPath, binPath);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

CNNNetwork Core::ReadNetwork(const std::string& model, const Blob::CPtr& weights) const {
    try {
        return _impl->ReadNetwork(model, weights);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network, const std::map<std::string, std::string>& config) {
    try {
        return LoadNetwork(network, ov::DEFAULT_DEVICE_NAME, config);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    auto valid = ::CheckStatic(network);
    try {
        OPENVINO_ASSERT(std::get<0>(valid),
                        "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                        "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                        std::get<1>(valid));
        auto exec = _impl->LoadNetwork(network, deviceName, config);
        return {exec._ptr, exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::LoadNetwork(const CNNNetwork& network,
                                    RemoteContext::Ptr context,
                                    const std::map<std::string, std::string>& config) {
    auto valid = ::CheckStatic(network);
    try {
        OPENVINO_ASSERT(std::get<0>(valid),
                        "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                        "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                        std::get<1>(valid));
        auto exec = _impl->LoadNetwork(network, std::dynamic_pointer_cast<RemoteContext>(context), config);
        return {exec._ptr, exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath,
                                    const std::string& deviceName,
                                    const std::map<std::string, std::string>& config) {
    try {
        auto exec = _impl->LoadNetwork(modelPath, deviceName, config, [](const CNNNetwork& network) {
            auto valid = ::CheckStatic(network);
            OPENVINO_ASSERT(std::get<0>(valid),
                            "InferenceEngine::Core::LoadNetwork doesn't support inputs having dynamic shapes. ",
                            "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                            std::get<1>(valid));
        });
        return {exec._ptr, exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::LoadNetwork(const std::string& modelPath, const std::map<std::string, std::string>& config) {
    try {
        return LoadNetwork(modelPath, ov::DEFAULT_DEVICE_NAME, config);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

RemoteContext::Ptr Core::CreateContext(const std::string& deviceName, const ParamMap& params) {
    try {
        return _impl->CreateContext(deviceName, params);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

RemoteContext::Ptr Core::GetDefaultContext(const std::string& deviceName) {
    if (deviceName.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support remote context";
    }
    if (deviceName.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support remote context";
    }
    if (deviceName.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support remote context";
    }
    try {
        return _impl->GetDefaultContext(deviceName);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::AddExtension(IExtensionPtr extension, const std::string& deviceName_) {
    if (deviceName_.find("HETERO") == 0) {
        IE_THROW() << "HETERO device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("MULTI") == 0) {
        IE_THROW() << "MULTI device does not support extensions. Please, set extensions directly to fallback devices";
    }
    if (deviceName_.find("AUTO") == 0) {
        IE_THROW() << "AUTO device does not support extensions. Please, set extensions directly to fallback devices";
    }

    try {
        _impl->AddExtension(extension);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::AddExtension(const IExtensionPtr& extension) {
    try {
        _impl->AddExtension(extension);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::ImportNetwork(const std::string& modelFileName,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::ImportNetwork");
    auto parsed = ov::parseDeviceNameIntoConfig(deviceName, ov::any_copy(config));
    std::ifstream modelStream(modelFileName, std::ios::binary);
    if (!modelStream.is_open())
        IE_THROW(NetworkNotRead) << "Model file " << modelFileName << " cannot be opened!";
    try {
        auto exec = _impl->get_plugin(parsed._deviceName).import_model(modelStream, parsed._config);
        return {ov::legacy_convert::convert_compiled_model(exec), exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::ImportNetwork");
    try {
        auto exec = _impl->ImportNetwork(networkModel, deviceName, config);
        return {exec._ptr, exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::ImportNetwork");

    using ExportMagic = std::array<char, 4>;
    constexpr static const ExportMagic exportMagic = {{0x1, 0xE, 0xE, 0x1}};

    std::string deviceName;
    ExportMagic magic = {};
    auto currentPos = networkModel.tellg();
    networkModel.read(magic.data(), magic.size());
    if (exportMagic == magic) {
        std::getline(networkModel, deviceName);
    } else {
        IE_THROW() << "Passed compiled stream does not contain device name. "
                      "Please, provide device name manually";
    }
    networkModel.seekg(currentPos, networkModel.beg);

    try {
        auto exec = _impl->get_plugin(deviceName).import_model(networkModel, {});
        return {ov::legacy_convert::convert_compiled_model(exec), exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

ExecutableNetwork Core::ImportNetwork(std::istream& networkModel,
                                      const RemoteContext::Ptr& context,
                                      const std::map<std::string, std::string>& config) {
    OV_ITT_SCOPED_TASK(ov::itt::domains::OV, "Core::ImportNetwork");

    if (context == nullptr) {
        IE_THROW() << "Remote context is null";
    }

    std::string deviceName_ = context->getDeviceName();
    ov::DeviceIDParser device(deviceName_);
    std::string deviceName = device.get_device_name();

    try {
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName, ov::any_copy(config));
        auto exec =
            _impl->get_plugin(deviceName)
                .import_model(networkModel, ov::legacy_convert::convert_remote_context(context), parsed._config);
        return {ov::legacy_convert::convert_compiled_model(exec), exec._so};
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

QueryNetworkResult Core::QueryNetwork(const CNNNetwork& network,
                                      const std::string& deviceName,
                                      const std::map<std::string, std::string>& config) const {
    try {
        auto valid = ::CheckStatic(network);
        OPENVINO_ASSERT(std::get<0>(valid),
                        "InferenceEngine::Core::QueryNetwork doesn't support inputs having dynamic shapes. ",
                        "Use ov::Core::compile_model API instead. Dynamic inputs are :",
                        std::get<1>(valid));

        return _impl->QueryNetwork(network, deviceName, config);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::SetConfig(const std::map<std::string, std::string>& config, const std::string& deviceName) {
    // HETERO case
    if (deviceName.find("HETERO:") == 0) {
        IE_THROW() << "SetConfig is supported only for HETERO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the HETERO on top.";
    }

    // MULTI case
    if (deviceName.find("MULTI:") == 0) {
        IE_THROW() << "SetConfig is supported only for MULTI itself (without devices). "
                      "You can configure the devices with SetConfig before creating the MULTI on top.";
    }

    // AUTO case
    if (deviceName.find("AUTO:") == 0) {
        IE_THROW() << "SetConfig is supported only for AUTO itself (without devices). "
                      "You can configure the devices with SetConfig before creating the AUTO on top.";
    }

    ov::AnyMap conf = ov::any_copy(config);
    try {
        if (deviceName.empty()) {
            _impl->set_property_for_device(conf, std::string());
        } else {
            _impl->set_property_for_device(conf, deviceName);
        }
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

Parameter Core::GetConfig(const std::string& deviceName, const std::string& name) const {
    // HETERO case
    {
        if (deviceName.find("HETERO:") == 0) {
            IE_THROW() << "You can only GetConfig of the HETERO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the HETERO on top.";
        }
    }
    // MULTI case
    {
        if (deviceName.find("MULTI:") == 0) {
            IE_THROW() << "You can only GetConfig of the MULTI itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the MULTI on top.";
        }
    }
    // AUTO case
    {
        if (deviceName.find("AUTO:") == 0) {
            IE_THROW() << "You can only GetConfig of the AUTO itself (without devices). "
                          "GetConfig is also possible for the individual devices before creating the AUTO on top.";
        }
    }

    if (name == CONFIG_KEY(FORCE_TBB_TERMINATE)) {
        const auto flag = ov::threading::executor_manager()->get_property(ov::force_tbb_terminate.name()).as<bool>();
        return flag ? CONFIG_VALUE(YES) : CONFIG_VALUE(NO);
    }

    try {
        auto parsed = ov::parseDeviceNameIntoConfig(deviceName);
        return _impl->get_plugin(parsed._deviceName).get_property(name, parsed._config);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

Parameter Core::GetMetric(const std::string& deviceName, const std::string& name, const ParamMap& options) const {
    try {
        return _impl->GetMetric(deviceName, name, options);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

std::vector<std::string> Core::GetAvailableDevices() const {
    try {
        return _impl->GetAvailableDevices();
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::RegisterPlugin(const std::string& pluginName, const std::string& deviceName) {
    try {
        _impl->register_plugin(pluginName, deviceName, {});
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::RegisterPlugins(const std::string& xmlConfigFile) {
    try {
        _impl->register_plugins_in_registry(xmlConfigFile);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what();
    }
}

void Core::UnregisterPlugin(const std::string& deviceName_) {
    try {
        ov::DeviceIDParser parser(deviceName_);
        std::string deviceName = parser.get_device_name();

        _impl->unload_plugin(deviceName);
    } catch (const ov::Exception& ex) {
        IE_THROW() << ex.what() << std::endl;
    }
}

}  // namespace InferenceEngine
