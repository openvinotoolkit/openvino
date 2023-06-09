// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "itt.hpp"
#include "plugin.hpp"
#include "compiled_model.hpp"

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>

#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/util/common_util.hpp"
#include "ie/ie_plugin_config.hpp"

#include "internal_properties.hpp"

// TODO (vurusovs) required for conversion to legacy API 1.0
#include "converter_utils.hpp"
#include "plugin.hpp"
// TODO (vurusovs) required for conversion to legacy API 1.0

ov::hetero::Plugin::Plugin() {
    set_device_name("HETERO");
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::compile_model");
    
    auto temp_properties(properties);
    auto config = Configuration{temp_properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(
        model->clone(),
        shared_from_this(),
        config);
    return compiled_model;

}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::RemoteContext& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
    const ov::RemoteContext& context,
    const ov::AnyMap& properties) const  {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::import_model");
    
    auto temp_properties(properties);
    // ov::hetero::CompiledModel compiled_model;
    return nullptr;
    // auto config = Configuration{device_properties, m_cfg};

    // read XML content
    // std::string xmlString;
    // std::uint64_t dataSize = 0;
    // model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    // xmlString.resize(dataSize);
    // model.read(const_cast<char*>(xmlString.c_str()), dataSize);

    // // read blob content
    // ov::Tensor weights;
    // model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    // if (0 != dataSize) {
    //     weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
    //     model.read(weights.data<char>(), dataSize);
    // }

    
    // std::string heteroXmlStr;
    // std::getline(heteroModel, heteroXmlStr);

    // pugi::xml_document heteroXmlDoc;
    // pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    // if (res.status != pugi::status_ok) {
    //     IE_THROW(NetworkNotRead) << "Error reading HETERO device xml header";
    // }

    // using namespace pugixml::utils;

    // pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    // compiled_model.m_name = GetStrAttr(heteroNode, "name");

    // std::unordered_set<std::string> networkInputs;
    // pugi::xml_node inputsNode = heteroNode.child("inputs");
    // FOREACH_CHILD (inputNode, inputsNode, "input") { networkInputs.insert(GetStrAttr(inputNode, "name")); }



    // std::unordered_set<std::string> networkOutputs;
    // pugi::xml_node outputsNode = heteroNode.child("outputs");
    // FOREACH_CHILD (outputNode, outputsNode, "output") { networkOutputs.insert(GetStrAttr(outputNode, "name")); }

    // auto heteroConfigsNode = heteroNode.child("hetero_config");
    // FOREACH_CHILD (heteroConfigNode, heteroConfigsNode, "config") {
    //     temp_properties.emplace(GetStrAttr(heteroConfigNode, "key"), GetStrAttr(heteroConfigNode, "value"));
    // }

    // auto deviceConfigsNode = heteroNode.child("device_config");
    // FOREACH_CHILD (deviceConfigNode, deviceConfigsNode, "config") {
    //     temp_properties.emplace(GetStrAttr(deviceConfigNode, "key"), GetStrAttr(deviceConfigNode, "value"));
    // }

    // // Erase all "hetero" properties from `temp_properties`
    // // to fill `m_cfg` and leave only properties for
    // // underlying devices
    // m_cfg = ov::hetero::Configuration(temp_properties, plugin->m_cfg);

    // auto blobNamesNode = heteroNode.child("blob_names_map");
    // FOREACH_CHILD (blobNameNode, blobNamesNode, "blob_name_map") {
    //     _blobNameMap.emplace(GetStrAttr(blobNameNode, "key"), GetStrAttr(blobNameNode, "value"));
    // }

    // std::vector<ov::hetero::CompiledModel::NetworkDesc> descs;
    // pugi::xml_node subnetworksNode = heteroNode.child("subnetworks");
    // FOREACH_CHILD (subnetworkNode, subnetworksNode, "subnetwork") {
    //     auto deviceName = GetStrAttr(subnetworkNode, "device");

    //     auto metaDevices = get_hetero_plugin()->get_properties_per_device(deviceName, temp_properties);
    //     assert(metaDevices.size() == 1);
    //     auto& loadConfig = metaDevices[deviceName];

    //     bool loaded = false;
    //     if (std::dynamic_pointer_cast<InferenceEngine::ICore>(plugin->get_core())
    //             ->DeviceSupportsModelCaching(deviceName)) {  // TODO (vurusovs) TEMPORARY SOLUTION
    //         compiled_model = plugin->get_core()->import_model(heteroModel, deviceName, loadConfig);
    //     } else {
    //         // read XML content
    //         std::string xmlString;
    //         std::uint64_t dataSize = 0;
    //         heteroModel.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    //         xmlString.resize(dataSize);
    //         heteroModel.read(const_cast<char*>(xmlString.c_str()), dataSize);

    //         /// read blob content
    //         ov::Tensor weights;
    //         heteroModel.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
    //         if (0 != dataSize) {
    //             weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
    //             heteroModel.read(weights.data<char>(), dataSize);
    //         }

    //         auto ov_model = plugin->get_core()->read_model(xmlString, weights);
    //         compiled_model = plugin->get_core()->compile_model(ov_model, deviceName, loadConfig);
    //         loaded = true;
    //     }

    //     // restore network inputs and outputs
    //     for (auto&& input : executableNetwork->GetInputsInfo()) {
    //         if (networkInputs.end() != networkInputs.find(input.first)) {
    //             _networkInputs.emplace(input.first, std::make_shared<InputInfo>(*input.second));
    //         }
    //     }

    //     for (auto&& output : executableNetwork->GetOutputsInfo()) {
    //         if (networkOutputs.end() != networkOutputs.find(output.first)) {
    //             _networkOutputs.emplace(output.first, std::make_shared<Data>(*output.second));
    //         }
    //     }

    //     compiled_model.m_networks.emplace_back(ov::hetero::CompiledModel::NetworkDesc{
    //         deviceName,
    //         loaded ? ov_model : ov::Model{},
    //         compiled_model,
    //     });
    // }
    // const auto parseNode = [](const pugi::xml_node& xml_node, bool is_param) -> std::shared_ptr<const ov::Node> {
    //     const std::string operation_name = GetStrAttr(xml_node, "operation_name");
    //     const auto elementType = ov::EnumNames<ov::element::Type_t>::as_enum(GetStrAttr(xml_node, "element_type"));

    //     std::vector<ov::Dimension> partialShape;
    //     pugi::xml_node partialShapeNode = xml_node.child("partial_shape");
    //     FOREACH_CHILD (dimNode, partialShapeNode, "dim") {
    //         partialShape.emplace_back(ov::Dimension(GetInt64Attr(dimNode, "value")));
    //     }

    //     pugi::xml_node tensorNamesNode = xml_node.child("tensor_names");
    //     std::unordered_set<std::string> tensorNames;
    //     FOREACH_CHILD (tensorNameNode, tensorNamesNode, "tensor_name") {
    //         tensorNames.insert(GetStrAttr(tensorNameNode, "value"));
    //     }

    //     std::shared_ptr<ov::Node> node = std::make_shared<ov::op::v0::Parameter>(elementType, partialShape);
    //     // For result operation_name is name of previous operation
    //     node->set_friendly_name(operation_name);
    //     if (!is_param)
    //         node = std::make_shared<ov::op::v0::Result>(node);
    //     node->output(0).get_tensor().add_names(tensorNames);

    //     return node;
    // };
    // (void)parseNode;

    // pugi::xml_node parametersNode = heteroNode.child("parameters");
    // FOREACH_CHILD (parameterNode, parametersNode, "parameter") {
    //     _parameters.emplace_back(parseNode(parameterNode, true));
    // }

    // pugi::xml_node resultsNode = heteroNode.child("results");
    // FOREACH_CHILD (resultNode, resultsNode, "result") { _results.emplace_back(parseNode(resultNode, false)); }

    // // save state
    // this->_networks = std::move(descs);
    // this->SetPointerToPlugin(_heteroPlugin->shared_from_this());
}

ov::hetero::Plugin::DeviceProperties ov::hetero::Plugin::get_properties_per_device(const std::string& device_priorities,
                                                                                   const ov::AnyMap& properties) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (auto&& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it) {
            device_properties[device_name] = get_core()->get_supported_property(device_name, properties);
        }
    }
    return device_properties;
}

ov::SupportedOpsMap ov::hetero::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::query_model");

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    auto device_properties = properties;
    Configuration config{device_properties, m_cfg};    
    DeviceProperties properties_per_device = get_properties_per_device(config.device_priorities, device_properties);

    std::map<std::string, ov::SupportedOpsMap> query_results;
    for (auto&& it : properties_per_device) {
        const auto& device_name = it.first;
        const auto& device_config = it.second;
        query_results[device_name] = get_core()->query_model(model, device_name, device_config);
    }

    //  WARNING: Here is devices with user set priority
    auto device_names = ov::DeviceIDParser::get_hetero_devices(config.device_priorities);

    ov::SupportedOpsMap res;
    for (auto&& device_name : device_names) {
        for (auto&& layer_query_result : query_results[device_name]) {
            res.emplace(layer_query_result);
        }
    }

    return res;
}

void ov::hetero::Plugin::set_property(const ov::AnyMap& properties) {
    auto temp_properties(properties);
    m_cfg = Configuration{temp_properties, m_cfg, true};
}

ov::Any ov::hetero::Plugin::get_property(const std::string& name, const ov::AnyMap& properties) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };

    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::caching_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities
                                                    };
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };

    auto temp_properties(properties);
    Configuration config{temp_properties, m_cfg};
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();

        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        add_ro_properties(METRIC_KEY(IMPORT_EXPORT_SUPPORT), metrics);
        return to_string_vector(metrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return to_string_vector(config.GetSupported());
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"HETERO"};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        return true;
    } else if (ov::caching_properties == name) {
        return decltype(ov::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        return caching_device_properties(config.device_priorities);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return config.Get(name);
    }
}

ov::Any ov::hetero::Plugin::caching_device_properties(const std::string& device_priorities) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    // Vector of caching properties per device
    std::vector<ov::AnyMap> result = {};
    for (const auto& device_name : device_names) {
        ov::AnyMap properties = {};
        auto supported_properties =
            get_core()->get_property(device_name, ov::supported_properties);
        if (ov::util::contains(supported_properties, ov::caching_properties)) {
            auto caching_properties =
                get_core()->get_property(device_name, ov::caching_properties);
            for (const auto& property_name : caching_properties) {
                properties[property_name] = get_core()->get_property(device_name, std::string(property_name), {});
            }
        } else if (ov::util::contains(supported_properties, ov::device::architecture)) {
            // If caching properties are not supported by device, try to add at least device architecture
            auto device_architecture = get_core()->get_property(device_name, ov::device::architecture);
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
        } else {
            // Device architecture is not supported, add device name w/o id as achitecture
            ov::DeviceIDParser parser(device_name);
            properties = ov::AnyMap{{ov::device::architecture.name(), parser.get_device_name()}};
        }
        result.emplace_back(properties);
    }
    return ov::Any(result);
}


std::shared_ptr<ov::IRemoteContext> ov::hetero::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> ov::hetero::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

static const ov::Version version = {CI_BUILD_NUMBER, "openvino_hetero_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::hetero::Plugin, version)