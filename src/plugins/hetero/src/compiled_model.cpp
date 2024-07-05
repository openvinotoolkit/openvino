// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "compiled_model.hpp"

#include <memory>

#include "async_infer_request.hpp"
#include "graph_debug_dump.hpp"
#include "itt.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/xml_parse_utils.hpp"
#include "plugin.hpp"
#include "properties.hpp"

ov::hetero::CompiledModel::CompiledModel(const std::shared_ptr<ov::Model>& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg)
    : ov::ICompiledModel(model, plugin),
      m_cfg(cfg),
      m_name(model->get_friendly_name()),
      m_loaded_from_cache(false) {
    try {
        compile_model(model);
    } catch (const std::exception& e) {
        OPENVINO_THROW("Standard exception from compilation library: ", e.what());
    } catch (...) {
        OPENVINO_THROW("Generic exception is thrown");
    }
}

void ov::hetero::CompiledModel::compile_model(const std::shared_ptr<ov::Model>& model) {
    ov::SupportedOpsMap query_model_result;
    bool user_set_affinities = false;
    // Get user defined affinity
    for (const auto& node : model->get_ordered_ops()) {
        auto& node_info = node->get_rt_info();
        auto it_info = node_info.find("affinity");
        if (it_info != node_info.end()) {
            OPENVINO_ASSERT(it_info->second.is<std::string>(), "Unexpected type of \"affinity\" attribute");
            query_model_result.emplace(node->get_friendly_name(), it_info->second.as<std::string>());
            user_set_affinities = true;
        }
    }

    auto compile_device_model = [&](CompiledModelDesc& compiled_model_desc, bool add_exclusive) {
        auto meta_devices =
            get_hetero_plugin()->get_properties_per_device(compiled_model_desc.device, m_cfg.get_device_properties());
        // disable caching for subgraphs, because the whole HETERO model is cached
        auto device_config = meta_devices[compiled_model_desc.device];
        device_config[ov::cache_dir.name()] = "";
        // set exclusive_async_requests in case when model is split
        if (add_exclusive) {
            auto supported_internal_properties =
                get_hetero_plugin()->get_core()->get_property(compiled_model_desc.device,
                                                              ov::internal::supported_properties);
            if (std::find(supported_internal_properties.begin(),
                          supported_internal_properties.end(),
                          ov::internal::exclusive_async_requests) != supported_internal_properties.end()) {
                // adds property if it is not set yet
                device_config.insert(ov::internal::exclusive_async_requests(true));
            }
        }
        compiled_model_desc.compiled_model = get_hetero_plugin()->get_core()->compile_model(compiled_model_desc.model,
                                                                                            compiled_model_desc.device,
                                                                                            device_config);
    };

    if (user_set_affinities) {
        // All affinities must be defined by user
        ov::hetero::SubgraphsVector ordered_subgraphs;
        std::tie(ordered_subgraphs, m_mapping_info) =
            get_model_subgraphs(model, query_model_result, user_set_affinities, m_cfg.dump_dot_files());

        m_compiled_submodels.resize(ordered_subgraphs.size());
        bool add_exclusive = ordered_subgraphs.size() > 1;
        size_t id = 0;
        for (const auto& subgraph : ordered_subgraphs) {
            m_compiled_submodels[id].device = subgraph._affinity;
            m_compiled_submodels[id].model = std::make_shared<ov::Model>(subgraph._results,
                                                                         subgraph._sinks,
                                                                         subgraph._parameters,
                                                                         m_name + '_' + std::to_string(id));
            compile_device_model(m_compiled_submodels[id], add_exclusive);
            ++id;
        }
    } else {
        // Restore properties in order to pass "device priorities" together
        // with devices properties
        auto full_properties = m_cfg.get_hetero_properties();
        for (const auto& property : m_cfg.get_device_properties())
            full_properties[property.first] = property.second;

        // This function modifes original model
        auto cloned_model = model->clone();
        std::tie(query_model_result, m_mapping_info) =
            get_hetero_plugin()->query_model_update(cloned_model, full_properties, true);

        ov::hetero::op::DeviceSubgraphVector ordered_subgraphs;
        for (const auto& op : cloned_model->get_ordered_ops()) {
            if (const auto& subgraph = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                ordered_subgraphs.push_back(subgraph);
            } else {
                OPENVINO_ASSERT(ov::op::util::is_output(op) || ov::op::util::is_parameter(op) ||
                                ov::op::util::is_sink(op));
            }
        }
        m_compiled_submodels.resize(ordered_subgraphs.size());
        bool add_exclusive = ordered_subgraphs.size() > 1;
        size_t id = 0;
        for (const auto& subgraph : ordered_subgraphs) {
            m_compiled_submodels[id].device = subgraph->get_affinity();
            m_compiled_submodels[id].model = subgraph->get_function();
            compile_device_model(m_compiled_submodels[id], add_exclusive);
            ++id;
        }
    }
    set_inputs_and_outputs();
}

ov::hetero::CompiledModel::CompiledModel(std::istream& model,
                                         const std::shared_ptr<const ov::IPlugin>& plugin,
                                         const Configuration& cfg,
                                         const bool loaded_from_cache)
    : ov::ICompiledModel(nullptr, plugin),
      m_cfg(cfg),
      m_name(),
      m_loaded_from_cache(loaded_from_cache) {
    std::string heteroXmlStr;
    std::getline(model, heteroXmlStr);

    pugi::xml_document heteroXmlDoc;
    pugi::xml_parse_result res = heteroXmlDoc.load_string(heteroXmlStr.c_str());

    if (res.status != pugi::status_ok)
        OPENVINO_THROW("Failed to read Hetero device xml header");

    using namespace ov::util::pugixml;

    pugi::xml_node heteroNode = heteroXmlDoc.document_element();
    m_name = get_str_attr(heteroNode, "name");

    ov::AnyMap properties;
    auto heteroConfigsNode = heteroNode.child("hetero_config");
    // clang-format off
    FOREACH_CHILD(heteroConfigNode, heteroConfigsNode, "config") {
        properties.emplace(get_str_attr(heteroConfigNode, "key"), get_str_attr(heteroConfigNode, "value"));
    }

    m_cfg = ov::hetero::Configuration(properties, m_cfg);

    pugi::xml_node subnetworksNode = heteroNode.child("compiled_submodels");
    FOREACH_CHILD(subnetworkNode, subnetworksNode, "compiled_submodel") {
        auto device = get_str_attr(subnetworkNode, "device");

        auto meta_devices = get_hetero_plugin()->get_properties_per_device(device, m_cfg.get_device_properties());
        assert(meta_devices.size() == 1);
        auto& loadConfig = meta_devices[device];

        ov::SoPtr<ov::ICompiledModel> compiled_model;
        std::shared_ptr<ov::Model> ov_model;

        if (get_plugin()->get_core()->device_supports_model_caching(device)) {
            compiled_model = plugin->get_core()->import_model(model, device, loadConfig);
        } else {
            // read XML content
            std::string xmlString;
            std::uint64_t dataSize = 0;
            model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            xmlString.resize(dataSize);
            model.read(const_cast<char*>(xmlString.c_str()), dataSize);

            /// read blob content
            ov::Tensor weights;
            model.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
            if (0 != dataSize) {
                weights = ov::Tensor(ov::element::from<char>(), ov::Shape{static_cast<ov::Shape::size_type>(dataSize)});
                model.read(weights.data<char>(), dataSize);
            }

            ov_model = plugin->get_core()->read_model(xmlString, weights);
            compiled_model = plugin->get_core()->compile_model(ov_model, device, loadConfig);
        }

        m_compiled_submodels.emplace_back(ov::hetero::CompiledModel::CompiledModelDesc{
            device,
            ov_model,
            compiled_model,
        });
    }

    auto inputs_map_node = heteroNode.child("inputs_to_submodels_inputs");
    FOREACH_CHILD(xml_node, inputs_map_node, "pair") {
        m_mapping_info._inputs_to_submodels_inputs.emplace_back(get_uint64_attr(xml_node, "submodel_idx"),
                                                  get_uint64_attr(xml_node, "node_idx"));
    }
    auto outputs_map_node = heteroNode.child("outputs_to_submodels_outputs");
    FOREACH_CHILD(xml_node, outputs_map_node, "pair") {
        m_mapping_info._outputs_to_submodels_outputs.emplace_back(get_uint64_attr(xml_node, "submodel_idx"),
                                                    get_uint64_attr(xml_node, "node_idx"));
    }
    auto submodels_input_to_prev_output_node = heteroNode.child("submodels_input_to_prev_output");
    FOREACH_CHILD(xml_node, submodels_input_to_prev_output_node, "record") {
        std::pair<uint64_t, uint64_t> in_pair = {get_uint64_attr(xml_node, "in_submodel_idx"),
                                                 get_uint64_attr(xml_node, "in_node_idx")};
        std::pair<uint64_t, uint64_t> out_pair = {get_uint64_attr(xml_node, "out_submodel_idx"),
                                                  get_uint64_attr(xml_node, "out_node_idx")};
        m_mapping_info._submodels_input_to_prev_output.emplace(in_pair, out_pair);
    }
    // clang-format on
    set_inputs_and_outputs();
}

std::shared_ptr<ov::ISyncInferRequest> ov::hetero::CompiledModel::create_sync_infer_request() const {
    return std::make_shared<ov::hetero::InferRequest>(
        std::static_pointer_cast<const ov::hetero::CompiledModel>(shared_from_this()));
}

std::shared_ptr<ov::IAsyncInferRequest> ov::hetero::CompiledModel::create_infer_request() const {
    auto internal_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<ov::hetero::AsyncInferRequest>(
        std::static_pointer_cast<ov::hetero::InferRequest>(internal_request),
        get_task_executor(),
        get_callback_executor());

    return async_infer_request;
}

void ov::hetero::CompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. "
                                   "Set property to Core::compile_model during compilation");
}

std::shared_ptr<const ov::Model> ov::hetero::CompiledModel::get_runtime_model() const {
    std::vector<std::shared_ptr<ov::Model>> rt_models;
    std::vector<std::shared_ptr<void>> shared_objects;
    // Collect runtime subgraphs
    rt_models.reserve(m_compiled_submodels.size());
    shared_objects.reserve(m_compiled_submodels.size());
    for (auto& compiled_submodel : m_compiled_submodels) {
        rt_models.push_back(compiled_submodel.compiled_model->get_runtime_model()->clone());
        shared_objects.push_back(compiled_submodel.compiled_model._so);
    }
    ov::hetero::merge_submodels(rt_models, m_mapping_info._submodels_input_to_prev_output);
    auto& runtime_graph = rt_models[0];
    OPENVINO_ASSERT(runtime_graph->inputs().size() == inputs().size());
    auto merged_shared_object = std::make_shared<std::vector<std::shared_ptr<void>>>(std::move(shared_objects));
    set_model_shared_object(
        *runtime_graph,
        std::shared_ptr<void>(std::move(merged_shared_object), reinterpret_cast<void*>(merged_shared_object.get())));
    return runtime_graph;
}

std::shared_ptr<const ov::hetero::Plugin> ov::hetero::CompiledModel::get_hetero_plugin() const {
    auto plugin = get_plugin();
    OPENVINO_ASSERT(plugin);
    auto hetero_plugin = std::static_pointer_cast<const ov::hetero::Plugin>(plugin);
    OPENVINO_ASSERT(hetero_plugin);
    return hetero_plugin;
}

ov::Any ov::hetero::CompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::execution_devices,
                                                    ov::loaded_from_cache,
                                                    ov::hetero::number_of_submodels};
        return ro_properties;
    };

    if (ov::supported_properties == name) {
        auto supported_properties = default_ro_properties();
        add_ro_properties(ov::supported_properties.name(), supported_properties);
        add_ro_properties(ov::device::properties.name(), supported_properties);
        add_ro_properties(ov::device::priorities.name(), supported_properties);
        return decltype(ov::supported_properties)::value_type(std::move(supported_properties));
    } else if (ov::device::properties == name) {
        ov::AnyMap all_devices = {};
        for (const auto& comp_model_desc : m_compiled_submodels) {
            ov::AnyMap device_properties = {};
            if (all_devices.count(comp_model_desc.device) == 0) {
                auto device_supported_props =
                    comp_model_desc.compiled_model->get_property(ov::supported_properties.name());
                for (auto&& property_name : device_supported_props.as<std::vector<ov::PropertyName>>())
                    device_properties[property_name] = comp_model_desc.compiled_model->get_property(property_name);
                all_devices[comp_model_desc.device] = device_properties;
            }
        }
        return all_devices;
    } else if (ov::model_name == name) {
        return decltype(ov::model_name)::value_type(m_name);
    } else if (ov::loaded_from_cache == name) {
        return decltype(ov::loaded_from_cache)::value_type{m_loaded_from_cache};
    } else if (ov::optimal_number_of_infer_requests == name) {
        unsigned int value = 0u;
        for (const auto& comp_model_desc : m_compiled_submodels) {
            value = std::max(value,
                             comp_model_desc.compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                 .as<unsigned int>());
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type{value};
    } else if (ov::execution_devices == name) {
        std::vector<std::string> device_names;
        std::set<std::string> s;
        for (const auto& comp_model_desc : m_compiled_submodels) {
            if (s.count(comp_model_desc.device) != 0)
                continue;
            s.insert(comp_model_desc.device);
            device_names.push_back(comp_model_desc.device);
        }
        return decltype(ov::execution_devices)::value_type{std::move(device_names)};
    } else if (ov::hetero::number_of_submodels == name) {
        return decltype(ov::hetero::number_of_submodels)::value_type{
            (m_compiled_submodels.size() - get_hetero_plugin()->independent_submodel_size)};
    }
    return m_cfg.get(name);
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::inputs() const {
    return m_compiled_inputs;
}

const std::vector<ov::Output<const ov::Node>>& ov::hetero::CompiledModel::outputs() const {
    return m_compiled_outputs;
}

void ov::hetero::CompiledModel::set_inputs_and_outputs() {
    // Restore inputs/outputs from compiled submodels
    m_compiled_inputs.reserve(m_mapping_info._inputs_to_submodels_inputs.size());
    for (const auto& it : m_mapping_info._inputs_to_submodels_inputs) {
        const auto& submodel_idx = it.first;
        OPENVINO_ASSERT(submodel_idx < m_compiled_submodels.size(),
                        "Model contains " + std::to_string(m_compiled_submodels.size()) +
                            " submodels. Index is out of range: " + std::to_string(submodel_idx));
        const auto& compiled_submodel = m_compiled_submodels[submodel_idx].compiled_model;
        const auto& input_idx = it.second;
        OPENVINO_ASSERT(input_idx < compiled_submodel->inputs().size(),
                        "Submodel " + std::to_string(submodel_idx) + " has " +
                            std::to_string(compiled_submodel->inputs().size()) +
                            " inputs. Index is out of range: " + std::to_string(input_idx));
        m_compiled_inputs.emplace_back(compiled_submodel->inputs()[input_idx]);
    }
    m_compiled_outputs.reserve(m_mapping_info._outputs_to_submodels_outputs.size());
    for (const auto& it : m_mapping_info._outputs_to_submodels_outputs) {
        const auto& submodel_idx = it.first;
        OPENVINO_ASSERT(submodel_idx < m_compiled_submodels.size(),
                        "Model contains " + std::to_string(m_compiled_submodels.size()) +
                            " submodels. Index is out of range: " + std::to_string(submodel_idx));
        const auto& compiled_submodel = m_compiled_submodels[submodel_idx].compiled_model;
        const auto& output_idx = it.second;
        OPENVINO_ASSERT(output_idx < compiled_submodel->outputs().size(),
                        "Submodel " + std::to_string(submodel_idx) + " has " +
                            std::to_string(compiled_submodel->outputs().size()) +
                            " outputs. Index is out of range: " + std::to_string(output_idx));
        m_compiled_outputs.emplace_back(compiled_submodel->outputs()[output_idx]);
    }
}

void ov::hetero::CompiledModel::export_model(std::ostream& model_stream) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "CompiledModel::export_model");

    pugi::xml_document doc;
    auto heteroNode = doc.append_child("hetero");
    heteroNode.append_attribute("name").set_value(m_name.c_str());

    auto inputs_map_node = heteroNode.append_child("inputs_to_submodels_inputs");
    for (const auto& it : m_mapping_info._inputs_to_submodels_inputs) {
        auto xml_node = inputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }
    auto outputs_map_node = heteroNode.append_child("outputs_to_submodels_outputs");
    for (const auto& it : m_mapping_info._outputs_to_submodels_outputs) {
        auto xml_node = outputs_map_node.append_child("pair");
        xml_node.append_attribute("submodel_idx").set_value(std::to_string(it.first).c_str());
        xml_node.append_attribute("node_idx").set_value(std::to_string(it.second).c_str());
    }

    auto submodels_input_to_prev_output_node = heteroNode.append_child("submodels_input_to_prev_output");
    for (const auto& it : m_mapping_info._submodels_input_to_prev_output) {
        auto xml_node = submodels_input_to_prev_output_node.append_child("record");
        xml_node.append_attribute("in_submodel_idx").set_value(std::to_string(it.first.first).c_str());
        xml_node.append_attribute("in_node_idx").set_value(std::to_string(it.first.second).c_str());
        xml_node.append_attribute("out_submodel_idx").set_value(std::to_string(it.second.first).c_str());
        xml_node.append_attribute("out_node_idx").set_value(std::to_string(it.second.second).c_str());
    }

    auto subnetworksNode = heteroNode.append_child("compiled_submodels");
    for (const auto& comp_model_desc : m_compiled_submodels) {
        auto subnetworkNode = subnetworksNode.append_child("compiled_submodel");
        subnetworkNode.append_attribute("device").set_value(comp_model_desc.device.c_str());
    }

    auto heteroConfigsNode = heteroNode.append_child("hetero_config");
    for (const auto& config : m_cfg.get_hetero_properties()) {
        auto heteroConfigNode = heteroConfigsNode.append_child("config");
        heteroConfigNode.append_attribute("key").set_value(config.first.c_str());
        heteroConfigNode.append_attribute("value").set_value(config.second.as<std::string>().c_str());
    }

    doc.save(model_stream, nullptr, pugi::format_raw);
    doc.reset();
    model_stream << std::endl;

    for (const auto& comp_model_desc : m_compiled_submodels) {
        OPENVINO_ASSERT(comp_model_desc.compiled_model);
        if (get_plugin()->get_core()->device_supports_model_caching(comp_model_desc.device)) {
            try {
                // Batch plugin reports property of low level plugin
                // If we use Batch plugin inside hetero, we won't be able to call export
                // Auto batch plugin will throw NOT_IMPLEMENTED
                comp_model_desc.compiled_model->export_model(model_stream);
                continue;
            } catch (ov::NotImplemented&) {
            }
        }
        auto& model = comp_model_desc.model;
        if (!model)
            OPENVINO_THROW("OpenVINO Model is empty");

        std::stringstream xmlFile, binFile;
        ov::pass::Serialize serializer(xmlFile, binFile);
        serializer.run_on_model(model);

        auto constants = binFile.str();
        auto model_str = xmlFile.str();

        auto dataSize = static_cast<std::uint64_t>(model_str.size());
        model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        model_stream.write(model_str.c_str(), dataSize);

        dataSize = static_cast<std::uint64_t>(constants.size());
        model_stream.write(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));
        model_stream.write(reinterpret_cast<char*>(&constants[0]), dataSize);
    }
}
